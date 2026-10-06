import numpy as np
import torch
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool  # multithreading
import logging

# Configure logging system
def setup_logging(log_level=logging.INFO):
    """Configure logging system"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Create global logger
logger = logging.getLogger(__name__)


def cal_hamming_dis(b1, b2):
    k = b2.shape[1]  # length of hash code
    dis = 0.5 * (k - np.dot(b1, b2.transpose()))
    return dis


def cal_cosine_dis(f1, f2):
    f1_norm = np.linalg.norm(f1)
    f2_norm = np.linalg.norm(f2, axis=1)

    similiarity = np.dot(f1, f2.T)/(f1_norm * f2_norm)
    return 1 - similiarity

def cal_map_gpu(query_feats, query_label, retrieval_feats, retrieval_label, top_k=5000, dist_method='hamming'):
    """
    Supports GPU acceleration + dynamic multithreading (automatically uses all CPUs), output identical to original function
    """
    query_number = query_label.shape[0]
    top_k_map = 0.0

    total_hit = 0
    hit_list = []
    last_hit_pos_list = []

    # ===================== Auto-detect whether to use GPU =====================
    use_gpu = torch.cuda.is_available()
    num_threads = os.cpu_count()  # dynamically get CPU core count
    # ================================================================

    # ---------------------- Utility functions ----------------------
    def cal_cosine_dis_torch(q, r):
        q_norm = q / (q.norm() + 1e-12)
        r_norm = r / (r.norm(dim=1, keepdim=True) + 1e-12)
        return 1 - torch.mm(q_norm, r_norm.T).squeeze()

    def cal_hamming_dis_torch(q, r):
        return torch.cdist(q.unsqueeze(0), r, p=0).squeeze()

    dist_func_torch = cal_hamming_dis_torch if dist_method == 'hamming' else cal_cosine_dis_torch

    # ---------------------- Single query computation (parallelizable) ----------------------
    def process_single_query(q_idx):
        q_feat = query_feats[q_idx:q_idx+1]
        q_lab = query_label[q_idx:q_idx+1]

        # ==== GPU transfer ====
        if use_gpu:

            q_feat = torch.cuda.FloatTensor(q_feat)
            q_lab = torch.cuda.FloatTensor(q_lab)
            r_feat = torch.cuda.FloatTensor(retrieval_feats)
            r_lab = torch.cuda.FloatTensor(retrieval_label)
        else:
            q_feat = torch.FloatTensor(q_feat)
            q_lab = torch.FloatTensor(q_lab)
            r_feat = torch.FloatTensor(retrieval_feats)
            r_lab = torch.FloatTensor(retrieval_label)

        # Distance
        dists = dist_func_torch(q_feat, r_feat)

        # Sort
        sort_idx = torch.argsort(dists)
        ground_truth = (torch.mm(q_lab, r_lab.T) > 0).squeeze().float()
        ground_truth_sorted = ground_truth[sort_idx]

        # topK
        top_k_gnd = ground_truth_sorted[:top_k]
        top_k_sum = int(top_k_gnd.sum().item())

        # Last hit position
        if top_k_sum > 0:
            hit_pos = (torch.where(top_k_gnd == 1)[0] + 1).cpu().numpy()
            last_hit_pos = int(hit_pos[-1])
        else:
            last_hit_pos = 0

        # AP
        if top_k_sum == 0:
            ap = 0.0
        else:
            pos = (torch.where(top_k_gnd == 1)[0] + 1).cpu().numpy()
            count = np.linspace(1, top_k_sum, top_k_sum)
            ap = np.mean(count / pos)

        return ap, top_k_sum, last_hit_pos

    # ===================== Dynamic threading: automatically use all CPU cores =====================
    pool = ThreadPool(num_threads)
    results = pool.map(process_single_query, range(query_number))
    pool.close()
    pool.join()
    # ======================================================================

    # Aggregate results
    for ap, top_k_sum, last_hit_pos in results:
        top_k_map += ap
        hit_list.append(top_k_sum)
        total_hit += top_k_sum
        last_hit_pos_list.append(last_hit_pos)

    # Average
    valid_last_hits = [x for x in last_hit_pos_list if x > 0]
    avg_last_hit_pos = np.mean(valid_last_hits) if len(valid_last_hits) > 0 else 0

    # Print
    logger.info(f"[cal_map] Total queries: {query_number} | top-k={top_k} | Using GPU: {use_gpu} | Threads: {num_threads}")
    # logger.info(f"[cal_map] Hits per query: {hit_list}")
    # logger.info(f"[cal_map] Last hit position per query: {last_hit_pos_list}")
    logger.info(f"[cal_map] Total hits: {total_hit} | Average hits: {total_hit / query_number:.2f}")
    logger.info(f"[cal_map] Average last hit position: {avg_last_hit_pos:.2f}")

    return top_k_map / query_number


def cal_map(query_feats, query_label, retrieval_feats, retrieval_label, top_k=5000, dist_method='hamming'):
    """
    Calculate MAP (Mean Average Precision)
    :param query_binary: binary code of query sample
    :param query_label: label of qurey sample
    :param retrieval_binary: binary code of database
    :param retrieval_label: label of database
    :param top_k:
    :return:
    """
    query_number = query_label.shape[0]
    top_k_map = 0

    # ===================== New: count hit numbers =====================
    total_hit = 0 # total hits across all queries
    hit_list = [] # save hit count of each query for printing
    last_hit_pos_list = []  # New: save last hit position of each query

    dist_func = cal_hamming_dis if dist_method == 'hamming' else cal_cosine_dis

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(np.float32)
        hamming_dis = dist_func(query_feats[query_index, :], retrieval_feats)

        sort_index = np.argsort(hamming_dis)
        ground_truth = ground_truth[sort_index]
        top_k_gnd = ground_truth[0:top_k]
        top_k_sum = np.sum(top_k_gnd).astype(int)

        # ===================== New: print current query hit count =====================
        hit_list.append(top_k_sum)
        total_hit += top_k_sum
        # ===================== New: get last hit position =====================
        if top_k_sum > 0:
            # Find all hit positions (starting from 1)
            hit_positions = np.where(top_k_gnd == 1)[0] + 1
            last_hit_pos = hit_positions[-1]  # last hit position
        else:
            last_hit_pos = 0  # no hit recorded as 0

        last_hit_pos_list.append(last_hit_pos)
        # ======================================================================

        if top_k_sum == 0:
            continue

        count = np.linspace(1, top_k_sum, int(top_k_sum))
        top_k_index = np.asarray(np.where(top_k_gnd == 1)) + 1.0
        top_k_map += np.mean(count / top_k_index)

    # Compute average last hit position
    valid_last_hits = [x for x in last_hit_pos_list if x > 0]
    avg_last_hit_pos = np.mean(valid_last_hits) if len(valid_last_hits) > 0 else 0

    # ===================== Print all information you want =====================
    logger.info(f"[cal_map] Total queries: {query_number} | top-k={top_k}")
    # logger.info(f"[cal_map] Hits per query: {hit_list}")
    # logger.info(f"[cal_map] Last hit position per query: {last_hit_pos_list}")
    logger.info(f"[cal_map] Total hits: {total_hit} | Average hits: {total_hit / query_number:.2f}")
    logger.info(f"[cal_map] Average last hit position: {avg_last_hit_pos:.2f}")
    # ==============================================================
    return top_k_map / query_number

def cal_pr(retrieval_binary, query_binary, retrieval_label, query_label, interval=0.1):
    r_arr = np.array([i * interval for i in range(1, int(1/interval) + 1)])
    p_arr = np.zeros(len(r_arr))

    query_number = query_label.shape[0]

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(
            np.float32)  # (1, N)
        hamming_dis = cal_hamming_dis(query_binary[query_index, :], retrieval_binary)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)
        ground_truth = ground_truth[sort_index]
        tp_num = len(np.where(ground_truth == 1)[0])
        r_num_arr = (tp_num * r_arr).astype(np.int32)

        tp_cum = np.cumsum(ground_truth)
        total_num_arr = np.array([np.where(tp_cum == i)[0][0] + 1 for i in r_num_arr])
        p_arr += r_num_arr/total_num_arr
    p_arr /= query_number

    return np.array(list(zip(r_arr, p_arr)))


def cal_top_n(retrieval_binary, query_binary, retrieval_label, query_label, top_n=None):
    if top_n is None:
        top_n = range(100, 1001, 100)

    top_n = np.array(top_n)
    top_n_p = np.zeros(len(top_n))
    query_number = query_label.shape[0]

    for query_index in range(query_number):
        ground_truth = (np.dot(query_label[query_index, :], retrieval_label.transpose()) > 0).astype(
            np.float32)  # (1, N)
        hamming_dis = cal_hamming_dis(query_binary[query_index, :], retrieval_binary)  # (1, N)

        # sort hamming distance
        sort_index = np.argsort(hamming_dis)
        ground_truth = ground_truth[sort_index]
        ground_truth = ground_truth[:top_n[-1]]

        tp_cum = np.cumsum(ground_truth)
        tp_num_arr = tp_cum[top_n - 1]
        top_n_p += tp_num_arr/top_n

    top_n_p /= query_number
    return np.array(list(zip(top_n, top_n_p)))