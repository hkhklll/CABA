import torch
import numpy as np
import pytorch_lightning as pl
import os

try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False
from abc import abstractmethod
from tqdm import tqdm
from utils.utils import FileLogger
from eval.metrics import cal_map
from eval.metrics import cal_map_gpu
from utils.utils import import_class, collect_outputs
from dataset.dataset import get_data_loader
from torch.utils.data import DataLoader
import time
import logging

# Configure tqdm to reduce refresh frequency (default 0.1s changed to 1s)
tqdm.monitor_interval = 0


class BaseCMR(pl.LightningModule):
    def __init__(self, cfg, binary=False) -> None:
        # ===================== New: configure logging timestamp format =====================
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # ==================================================================
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.binary = binary

        # init file logger
        self.flogger = FileLogger('log', '{}.log'.format(cfg['save_name']))
        self.flogger.log("=> Starting run {} ...".format(cfg['module_name']))
        # cache for FAISS indexes to avoid rebuilding index on every evaluation call
        self._faiss_index_cache = {}
        # New: database feature cache (initialized to None; during one test there are two tests with poisoned data and clean data, which will load database features twice, so add cache to reuse the first load in the second)
        self._db_feature_cache = None  # cache format: (db_img, db_txt, db_img_label, db_txt_label)

        # load model
        self.tokenizer, self.global_vectors, self.model = self.load_model()

        if self.global_vectors is None:
            self.vectorize_func = self.vectorize_batch_bert
        else:
            self.vectorize_func = self.vectorize_batch

        # load data
        if cfg['percentage'] > 0:
            self.poi_train_loader, self.poi_test_loader = self.load_poi_data()
            self.train_loader, self.test_loader, self.database_loader = self.load_data(load_train=False)
        else:
            self.train_loader, self.test_loader, self.database_loader = self.load_data(load_train=True)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def training_epoch_end(self, outputs):
        pass

    def load_poi_data(self):
        cfg = self.cfg
        attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
        attack = import_class(attack_method)(cfg)

        poi_train_dataset = attack.get_poisoned_data('train', p=cfg['percentage'])
        poi_train_loader = DataLoader(poi_train_dataset, batch_size=self.cfg['batch_size'], shuffle=True,
                                      num_workers=16, collate_fn=self.vectorize_func)

        poi_test_dataset = attack.get_poisoned_data('test', p=cfg.get('test_percentage', 1.0))
        poi_test_loader = DataLoader(poi_test_dataset, batch_size=self.cfg['batch_size'], shuffle=False,
                                     num_workers=16, collate_fn=self.vectorize_func)

        return poi_train_loader, poi_test_loader

    def load_data(self, load_train=True):
        cfg = self.cfg
        test_percentage = cfg.get('test_percentage', 1.0)

        if load_train:
            train_loader, _ = get_data_loader(
                cfg['data_path'], cfg['dataset'], 'train', batch_size=cfg['batch_size'],
                shuffle=True, collate_fn=self.vectorize_func)
        else:
            train_loader = None

        test_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'test', batch_size=cfg['batch_size'],
            shuffle=False, collate_fn=self.vectorize_func, test_percentage=test_percentage)

        database_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'database', batch_size=cfg['batch_size'],
            shuffle=False, collate_fn=self.vectorize_func)

        return train_loader, test_loader, database_loader

    def vectorize_batch(self, batch, max_length=40):
        img, text, img_label, txt_label, index = zip(*batch)
        img_tensor = torch.stack(img)
        img_label_tensor = torch.stack(img_label)
        txt_label_tensor = torch.stack(txt_label)

        text_embedding = []
        for t in text:
            tokens = self.tokenizer(t)
            tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
            text_embedding.append(self.global_vectors.get_vecs_by_tokens(tokens))
        text_tensor = torch.stack(text_embedding)

        return img_tensor, text_tensor, img_label_tensor, txt_label_tensor, index

    def vectorize_batch_bert(self, batch, max_length=40):
        img, text, img_label, txt_label, index = zip(*batch)
        img_tensor = torch.stack(img)
        img_label_tensor = torch.stack(img_label)
        txt_label_tensor = torch.stack(txt_label)

        tokens = self.tokenizer(list(text), max_length=max_length, padding='longest',
                                truncation='longest_first', return_tensors="pt")
        text_tensor = torch.cat((tokens.input_ids, tokens.attention_mask), dim=1)

        return img_tensor, text_tensor, img_label_tensor, txt_label_tensor, index

    def validation_step(self, batch, batch_idx):
        img, text, img_label, txt_label = batch[:4]
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(),
            "txt_feature": txt_feats.cpu().numpy(),
            "img_label": img_label.cpu().numpy(),
            "txt_label": txt_label.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        """
        retrieve on train_loader for fast validation
        """
        key_list = ['img_feature', 'txt_feature', 'img_label', 'txt_label']
        img_feats, txt_feats, img_label, txt_label = collect_outputs(outputs, key_list)
        img_feats, txt_feats, img_label, txt_label = np.concatenate(img_feats), np.concatenate(
            txt_feats), np.concatenate(img_label), np.concatenate(txt_label)

        img2txt = self.get_map_value(img_feats, img_label, txt_feats, txt_label)
        txt2img = self.get_map_value(txt_feats, txt_label, img_feats, img_label)
        self.flogger.log("=> img2txt MAP: {:.4f}  txt2img MAP: {:.4f}".format(img2txt, txt2img))
        val_map = (img2txt + txt2img) / 2
        self.log('val_map', value=val_map, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, text, img_label, txt_label = batch[:4]
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(),
            "txt_feature": txt_feats.cpu().numpy(),
            "img_label": img_label.cpu().numpy(),
            "txt_label": txt_label.cpu().numpy()}

    def test_epoch_end(self, outputs):
        # collect outputs of test_loader
        key_list = ['img_feature', 'txt_feature', 'img_label', 'txt_label']
        test_img, test_txt, test_img_label, test_txt_label = collect_outputs(outputs, key_list)
        test_img, test_txt, test_img_label, test_txt_label = np.concatenate(test_img), np.concatenate(
            test_txt), np.concatenate(test_img_label), np.concatenate(test_txt_label)

        # Bounded-read cache: only generate if no cache
        if self._db_feature_cache is None:
            self.flogger.log("=> Generating database features")
            start_time = time.time()
            db_img, db_txt, db_img_label, db_txt_label = self.generate_feature(self.model, self.database_loader)
            elapsed_time = time.time() - start_time
            # Convert to minutes and seconds
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            self.flogger.log(f"\n=> Database feature generation complete, elapsed: {minutes} min {seconds:.2f} sec")
            # Cache features
            self._db_feature_cache = db_img, db_txt, db_img_label, db_txt_label
        else:
            self.flogger.log("\n=> Reusing cached database features (skipping regeneration to save time)")
            db_img, db_txt, db_img_label, db_txt_label = self._db_feature_cache

        img2txt = self.get_map_value(test_img, test_img_label, db_txt, db_txt_label)
        txt2img = self.get_map_value(test_txt, test_txt_label, db_img, db_img_label)
        benign_accuracy = (img2txt + txt2img) / 2
        self.flogger.log("=> Query count: {}, Database size: {}".format(len(test_img_label), len(db_img_label)))
        self.flogger.log("=> MAP(img2txt): {:.4f}  MAP(txt2img): {:.4f}".format(img2txt, txt2img))
        self.flogger.log("=> BA: {:.4f}".format(benign_accuracy))

    @staticmethod
    def generate_feature(model, data_loader):
        model = model.eval()
        img_list, txt_list, img_label_list, txt_label_list = [], [], [], []
        for batch in tqdm(data_loader, desc="Generating features", mininterval=5):
            img, text, img_label, txt_label = batch[:4]

            img, text = img.cuda(), text.cuda()
            img_feats, txt_feats = model.inference(img, text)

            img_list.append(img_feats.cpu().numpy())
            txt_list.append(txt_feats.cpu().numpy())
            img_label_list.append(img_label.numpy())
            txt_label_list.append(txt_label.numpy())

        ret = (
            np.concatenate(img_list),
            np.concatenate(txt_list),
            np.concatenate(img_label_list),
            np.concatenate(txt_label_list)
        )
        return ret

    def get_map_value(self, query_feats, query_label, retrieval_feats, retrieval_label,
                      top_k=5000):
        # Entry log: record current retrieval settings to help troubleshoot issues of not entering FAISS branch
        self.flogger.log(
            f"=> Entering get_map_value: binary={self.binary}, top_k={top_k}, retrieval_size={retrieval_feats.shape[0]}")

        if self.binary:
            query_feats = np.sign(query_feats)
            retrieval_feats = np.sign(retrieval_feats)
            dist_method = 'hamming'
        else:
            dist_method = 'cosine'
        # If available, use FAISS for candidate retrieval and exact reranking to accelerate retrieval
        use_faiss = self.cfg.get('use_faiss', True) and HAS_FAISS
        if use_faiss and dist_method == 'cosine':
            # candidate_factor controls how many candidates to retrieve per query before reranking (supports decimals)
            candidate_factor = float(self.cfg.get('faiss_candidate_factor', 1.2))
            candidate_k = min(retrieval_feats.shape[0], max(int(top_k * candidate_factor), top_k + 100))
            # Whether to use GPU index (requires faiss-gpu installed and 'faiss_use_gpu': True in cfg)
            use_faiss_gpu = bool(self.cfg.get('faiss_use_gpu', False))
            if use_faiss_gpu:
                cfg_device = self.cfg.get('device', 0)
                if isinstance(cfg_device, (list, tuple)) and len(cfg_device) > 0:
                    gpu_id = int(cfg_device[0])
                else:
                    gpu_id = int(cfg_device)
            else:
                gpu_id = 0
            self.flogger.log(
                f"=> FAISS candidate retrieval start: use_gpu={use_faiss_gpu}, gpu_id={gpu_id}, candidate_k={candidate_k}, retrieval set size={retrieval_feats.shape[0]}")
            cand_indices = self._faiss_search_candidates(query_feats, retrieval_feats, candidate_k,
                                                         use_gpu=use_faiss_gpu, gpu_id=gpu_id)
            return self._calc_map_from_candidates(query_feats, query_label, retrieval_feats, retrieval_label,
                                                  cand_indices, candidate_k, dist_method)

        # fallback to original exact cal_map
        return cal_map(query_feats, query_label, retrieval_feats, retrieval_label, top_k, dist_method)

    def _faiss_search_candidates(self, query_feats, retrieval_feats, candidate_k, use_gpu=False, gpu_id=0):
        """
        Build FAISS index (use IndexFlatIP on normalized vectors to achieve cosine similarity effect),
        and return candidate indices for each query. Returned array shape is (Q, candidate_k).
        Supports index caching and optional GPU index (via parameters use_gpu, gpu_id).
        """
        # Force float32 and ensure C-contiguity
        q = np.ascontiguousarray(query_feats.astype('float32', copy=False))
        r = np.ascontiguousarray(retrieval_feats.astype('float32', copy=False))

        # Entry log
        self.flogger.log(
            f"=> Entering FAISS candidate retrieval: q_shape={q.shape}, r_shape={r.shape}, candidate_k={candidate_k}, use_gpu={use_gpu}, gpu_id={gpu_id}")

        # Prefer faiss built-in normalization (in-place, keep float32), otherwise fall back to numpy implementation
        try:
            faiss.normalize_L2(r)
            faiss.normalize_L2(q)
        except Exception:
            # numpy normalization (result still float32)
            r_norms = np.linalg.norm(r, axis=1, keepdims=True)
            r_norms[r_norms == 0] = 1.0
            r = (r / r_norms).astype('float32', copy=False)
            q_norms = np.linalg.norm(q, axis=1, keepdims=True)
            q_norms[q_norms == 0] = 1.0
            q = (q / q_norms).astype('float32', copy=False)

        dim = r.shape[1]

        # Try reading GPU configuration from cfg (cfg takes priority)
        cfg_use_gpu = bool(self.cfg.get('faiss_use_gpu', False))
        if cfg_use_gpu:
            cfg_device = self.cfg.get('device', 0)
            if isinstance(cfg_device, (list, tuple)) and len(cfg_device) > 0:
                cfg_gpu_id = int(cfg_device[0])
            else:
                # Support case of string '0'
                cfg_gpu_id = int(cfg_device)
        else:
            cfg_gpu_id = int(self.cfg.get('faiss_gpu_id', gpu_id)) if 'faiss_gpu_id' in self.cfg else gpu_id
        use_gpu = cfg_use_gpu or use_gpu
        gpu_id = cfg_gpu_id if cfg_use_gpu else gpu_id

        # build cache key and try to reuse index if available
        cache_key = ('faiss_index', id(retrieval_feats), dim, bool(use_gpu), int(gpu_id))
        idx = None
        if hasattr(self, '_faiss_index_cache') and cache_key in self._faiss_index_cache:
            idx = self._faiss_index_cache[cache_key]

        if idx is None:
            # Log before building index
            if use_gpu:
                self.flogger.log(f"=> Building FAISS GPU index: gpu_id={gpu_id}, dim={dim}, vector count={r.shape[0]}")
            else:
                self.flogger.log(f"=> Building FAISS CPU index: dim={dim}, vector count={r.shape[0]}")

            index = faiss.IndexFlatIP(dim)
            if use_gpu:
                try:
                    # create GPU resources and transfer index
                    if not hasattr(self, '_faiss_gpu_res') or self._faiss_gpu_res is None:
                        self._faiss_gpu_res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(self._faiss_gpu_res, int(gpu_id), index)
                except Exception as e:
                    # If GPU transfer fails, fall back to CPU index and log
                    self.flogger.log(f"=> Warning: failed to use FAISS GPU index: {e}. Falling back to CPU index.")
                    index = faiss.IndexFlatIP(dim)
            index.add(r)  # faiss will copy data into the index
            # cache index for reuse
            if not hasattr(self, '_faiss_index_cache'):
                self._faiss_index_cache = {}
            self._faiss_index_cache[cache_key] = index
            idx = index
        else:
            self.flogger.log(f"=> Reusing cached FAISS index: cache_key={cache_key}")

        distances, indices = idx.search(q, int(candidate_k))
        self.flogger.log(f"=> FAISS retrieval complete: returned index shape={indices.shape}")
        return indices

    def _calc_map_from_candidates(self, query_feats, query_label, retrieval_feats, retrieval_label,
                                  candidate_indices, top_k, dist_method):
        """
        Perform exact reranking on the candidate subset of each query and compute MAP.
        Reranking uses the same cosine distance metric as the original implementation, ensuring the reranked result is consistent with the original algorithm.
        """
        num_queries = query_label.shape[0]
        top_k_map = 0.0

        # ===================== New: count hit numbers & last hit position =====================
        total_hit = 0
        hit_list = []
        last_hit_pos_list = []
        # ==========================================================================

        for qi in range(num_queries):
            cand_idx = candidate_indices[qi]
            cand_feats = retrieval_feats[cand_idx]
            cand_labels = retrieval_label[cand_idx]

            # Compute cosine distance: 1 - cosine similarity
            qf = query_feats[qi:qi + 1].astype('float32', copy=False)
            # vectorized cosine similarity
            q_norm = np.linalg.norm(qf)
            c_norm = np.linalg.norm(cand_feats, axis=1)
            denom = (q_norm * c_norm)
            denom[denom == 0] = 1.0
            sims = np.dot(qf, cand_feats.T).reshape(-1) / denom
            dists = 1.0 - sims
            # Compute ground truth for candidates (1 if sharing any label)
            ground_truth = (np.dot(query_label[qi, :], cand_labels.T) > 0).astype(np.float32)

            # Sort by ascending distance
            sort_idx = np.argsort(dists)
            ground_truth_sorted = ground_truth[sort_idx]

            top_k_gnd = ground_truth_sorted[0:top_k]
            top_k_sum = int(np.sum(top_k_gnd))

            # ===================== New: record hit count =====================
            hit_list.append(top_k_sum)
            total_hit += top_k_sum
            # ============================================================

            # ===================== New: record last hit position =====================
            if top_k_sum > 0:
                hit_positions = np.where(top_k_gnd == 1)[0] + 1
                last_hit_pos = hit_positions[-1]
            else:
                last_hit_pos = 0
            last_hit_pos_list.append(last_hit_pos)
            # ==================================================================

            if top_k_sum == 0:
                continue

            count = np.linspace(1, top_k_sum, int(top_k_sum))
            top_k_index = np.asarray(np.where(top_k_gnd == 1)) + 1.0
            top_k_map += np.mean(count / top_k_index)

        # ===================== New: compute average and print =====================
        valid_last_hits = [x for x in last_hit_pos_list if x > 0]
        avg_last_hit_pos = np.mean(valid_last_hits) if len(valid_last_hits) > 0 else 0

        self.flogger.log(f"[_calc_map_from_candidates] Total queries: {num_queries} | top-k={top_k}")
        # self.flogger.log(f"[_calc_map_from_candidates] Hits per query: {hit_list}")
        # self.flogger.log(f"[_calc_map_from_candidates] Last hit position per query: {last_hit_pos_list}")
        self.flogger.log(
            f"[_calc_map_from_candidates] Total hits: {total_hit} | Average hits: {total_hit / num_queries:.2f}")
        self.flogger.log(f"[_calc_map_from_candidates] Average last hit position: {avg_last_hit_pos:.2f}")
        # ==================================================================

        return top_k_map / num_queries
