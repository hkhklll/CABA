# Text Keyword Extractor
# Use CLIP model to identify important words in text that are critical for cross-modal alignment
# Based on comprehensive similarity analysis: cross-modal similarity change + text semantic change

import os
import re
import argparse
import pickle
import json
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from dataset.dataset import CrossModalDataset
from dataset.dataset import get_dataset_filename


class TextKeywordExtractor:
    """
    Text Keyword Extractor

    Core functions:
    1. Mask each word (delete or replace with [MASK])
    2. Compute cross-modal similarity change (original text-image vs masked text-image)
    3. Compute text similarity (masked text vs original text)
    4. Comprehensive sensitivity score, select key words

    Theoretical basis:
    - Cross-modal alignment: text-image similarity
    - Text semantics: text encoding similarity
    """

    def __init__(self, args):
        """
        Initialize Text Keyword Extractor

        Args:
        - args: command line arguments, including model configuration, dataset paths, etc.
        """
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.max_text_len = args.max_text_len
        # Get subset ratio from command line arguments or default value (used to select data volume)
        self.ratio = getattr(args, 'ratio', 0.01)
        # Generate timestamp for this run (used for result file naming)
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        print(f"Using device: {self.device}")
        print(f"Max text length: {self.max_text_len}")

        # Load CLIP model
        self.clip_model, self.clip_processor = self._load_clip_model()
        self.clip_model.to(self.device)
        self.clip_model.eval()

        # Determine if CLIP supports directly calling vision tower / text tower (transformers 4.20+)
        self._clip_has_split = (hasattr(self.clip_model, 'get_image_features')
                                and hasattr(self.clip_model, 'get_text_features'))
        if not self._clip_has_split:
            print("  [Note] transformers version is old, will fall back to full forward() (slightly slower)")

        print("CLIP model loaded")

        # Load dataset
        self.dataset = self._load_dataset()
        print(f"Dataset loaded: {args.dataset} - {args.split}")

        # Stop words list
        self.stop_words = self._get_stop_words()

        # Text keyword extraction parameters
        self.text_mask_method = getattr(args, 'text_mask_method', 'merge')  # 'delete', 'mask' or 'merge'
        self.cross_modal_weight = getattr(args, 'cross_modal_weight', 0.5)  # cross-modal similarity weight
        self.words_thred = getattr(args, 'words_thred', 8)  # keyword count threshold
        self.filter_nouns_only = getattr(args, 'filter_nouns_only', True)  # whether to keep only nouns

        print(f"Text keyword extraction parameters:")
        print(f"  Mask method: {self.text_mask_method}")
        print(f"  Cross-modal similarity weight: {self.cross_modal_weight:.2f}")
        print(f"  Text similarity weight: {(1 - self.cross_modal_weight):.2f}")
        print(f"  Keyword count threshold: {self.words_thred}")
        print(f"  Keep only nouns: {self.filter_nouns_only}")

    def _load_clip_model(self):
        """
        Load CLIP model and processor

        Returns:
        - model: CLIP model
        - processor: CLIP processor
        """
        model = CLIPModel.from_pretrained(self.args.from_pretrain)
        processor = CLIPProcessor.from_pretrained(self.args.from_pretrain)
        model.eval()
        return model, processor

    def _load_dataset(self):
        """
        Load cross-modal dataset
        """
        from concurrent.futures import ThreadPoolExecutor

        transform = transforms.Compose([lambda x: np.array(x)])
        img_name, text_name, label_name = get_dataset_filename(self.args.split)
        full_dataset = CrossModalDataset(
            os.path.join(self.args.data_path, self.args.dataset),
            img_name, text_name, label_name, transform
        )

        # Only use the first ratio portion of data (adjustable via --ratio)
        subset_size = int(len(full_dataset) * self.ratio)
        subset = torch.utils.data.Subset(full_dataset, range(subset_size))

        # Preload all images to memory (multi-threaded parallel disk read), later directly read numpy in loop
        print(f"  Preloading {subset_size} images to memory (multi-threaded)...")
        # Image root directory = data_path + dataset name (CrossModalDataset also parses this way)
        data_root = os.path.join(self.args.data_path, self.args.dataset)

        def _load_one(idx):
            img_filename = full_dataset.imgs[idx]
            img = Image.open(os.path.join(data_root, img_filename)).convert('RGB')
            return np.array(img)

        with ThreadPoolExecutor(max_workers=min(16, subset_size)) as ex:
            images = list(tqdm(ex.map(_load_one, range(subset_size)),
                               total=subset_size, desc="Preloading images"))

        # Wrap into custom ListDataset (no longer accesses disk)
        class _ListDataset(torch.utils.data.Dataset):
            def __init__(self, images, dataset_subset, full_dataset):
                self.images = images
                self.subset = dataset_subset
                self.full_dataset = full_dataset

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, local_idx):
                original_idx = self.subset.indices[local_idx]
                return self.images[original_idx], self.full_dataset.texts[original_idx], original_idx

        wrapped = _ListDataset(images, subset, full_dataset)
        print(f"Full dataset size: {len(full_dataset)}, used data: {len(wrapped)}")
        return wrapped

    def _get_stop_words(self):
        """Get stop words list"""
        return {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "don", "should", "now", "foreground", "background"
        }

    def _is_likely_noun(self, word):
        """
        Use NLTK POS tagging to determine if a word is a noun (with LRU cache)

        Args:
        - word: word string

        Returns:
        - bool: whether it is a noun
        """
        if not self.filter_nouns_only:
            return True

        if not word or not isinstance(word, str):
            return False

        # Clean word and tokenize
        word_clean = re.sub(r'[^\w\s]', '', word.strip()).lower()
        if not word_clean:
            return False

        # Cache lookup
        if hasattr(self, '_noun_cache') and word_clean in self._noun_cache:
            return self._noun_cache[word_clean]

        try:
            tokens = word_tokenize(word_clean)
            if not tokens:
                if not hasattr(self, '_noun_cache'):
                    self._noun_cache = {}
                self._noun_cache[word_clean] = False
                return False
            tags = pos_tag(tokens)
            # If the first token is tagged as a noun, consider the word a noun
            tag = tags[0][1]
            result = tag in {'NN', 'NNS', 'NNP', 'NNPS'}
        except Exception:
            # Fall back to simple length check on error
            result = len(word_clean) > 1

        if not hasattr(self, '_noun_cache'):
            self._noun_cache = {}
        self._noun_cache[word_clean] = result
        return result

    def _create_masked_text(self, words, word_idx, use_delete=True):
        """
        Create masked text

        Args:
        - words: list of words
        - word_idx: index of word to mask
        - use_delete: True=delete directly, False=replace with [MASK]

        Returns:
        - masked_text: masked text
        """
        if use_delete:
            # Method 1: delete directly
            masked_words = words.copy()
            del masked_words[word_idx]
            return ' '.join(masked_words) if masked_words else ''
        else:
            # Method 2: replace with [MASK]
            masked_words = words.copy()
            masked_words[word_idx] = '[MASK]'
            return ' '.join(masked_words)

    def _safe_image_features(self, image):
        """
        Get image embedding (compatible with older transformers: directly use full forward then take image_embeds)

        Args:
        - image: PIL Image

        Returns:
        - img_embeds: tensor (1, embed_dim)
        """
        inputs = self.clip_processor(images=image, return_tensors="pt")
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        with torch.no_grad():
            if self._clip_has_split:
                return self.clip_model.get_image_features(**inputs)
            # Fallback: use processor with empty text '' placeholder, then take image_embeds from outputs
            fallback = self.clip_processor(
                text=[''], images=image, return_tensors="pt",
                max_length=self.max_text_len, padding='longest', truncation='longest_first'
            )
            for key, val in fallback.items():
                fallback[key] = val.to(self.device)
            outputs = self.clip_model(**fallback)
            return outputs.image_embeds

    def _safe_text_features(self, texts):
        """
        Get text embedding (compatible with older transformers)

        Args:
        - texts: string or list of strings

        Returns:
        - text_embeds: tensor (N, embed_dim)
        """
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        inputs = self.clip_processor(
            text=text_list, return_tensors="pt",
            max_length=self.max_text_len, padding='longest', truncation='longest_first'
        )
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        with torch.no_grad():
            if self._clip_has_split:
                embeds = self.clip_model.get_text_features(**inputs)
            else:
                # Fallback: use processor with empty image + text
                dummy_image = Image.new('RGB', (224, 224), color=0)
                fallback = self.clip_processor(
                    text=text_list, images=dummy_image, return_tensors="pt",
                    max_length=self.max_text_len, padding='longest', truncation='longest_first'
                )
                for key, val in fallback.items():
                    fallback[key] = val.to(self.device)
                outputs = self.clip_model(**fallback)
                embeds = outputs.text_embeds
            return embeds[0:1] if single else embeds

    def _compute_cross_modal_similarity(self, text, image, image_embeds=None):
        """
        Compute text-image cross-modal similarity

        Args:
        - text: text string
        - image: PIL Image or numpy array
        - image_embeds: optional, precomputed image embedding (for reuse)

        Returns:
        - similarity: cosine similarity value
        - text_embeds: text encoding (for subsequent computation)
        """
        # Convert image to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Prepare inputs
        inputs = self.clip_processor(
            text=text,
            images=image,
            return_tensors="pt",
            max_length=self.max_text_len,
            padding='longest',
            truncation='longest_first'
        )
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)

        # Compute similarity
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            if image_embeds is None:
                image_embeds = outputs.image_embeds
            similarity = F.cosine_similarity(
                image_embeds,
                outputs.text_embeds
            ).item()
            text_embeds = outputs.text_embeds

        return similarity, text_embeds

    def _compute_batch(self, texts, image, image_embeds=None):
        """
        Batch compute text-image cross-modal similarity and text encoding

        Args:
        - texts: list of text strings (length N)
        - image: PIL Image or numpy array
        - image_embeds: optional, precomputed image embedding

        Returns:
        - similarities: list of cosine similarities (length N)
        - text_embeds: text encoding tensor (N, embed_dim)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.clip_processor(
            text=texts,
            images=image,
            return_tensors="pt",
            max_length=self.max_text_len,
            padding='longest',
            truncation='longest_first'
        )
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            if image_embeds is None:
                image_embeds = outputs.image_embeds
            else:
                # image_embeds may be (1, dim) — broadcast to (N, dim)
                image_embeds = image_embeds.expand(len(texts), -1)
            similarities = F.cosine_similarity(
                image_embeds,
                outputs.text_embeds,
                dim=1
            )
            return similarities.cpu().numpy(), outputs.text_embeds

    def identify_critical_text_words(self, image, text, use_delete=None, _shared=None):
        """
        Identify critical text words - based on comprehensive similarity analysis (batch accelerated version)

        Method:
        1. Collect all candidate words, batch compute cross-modal similarities and text encodings for all masked texts at once
        2. Compute cross-modal similarity change (original text-image vs masked text-image)
        3. Compute text similarity (masked text vs original text)
        4. Comprehensive sensitivity = cross_modal_weight * cross_modal_change + (1-cross_modal_weight) * (1 - text_similarity)

        Args:
        - image: input image
        - text: input text
        - use_delete: True=delete directly, False=replace with [MASK], None=use default configuration
        - _shared: dict, optional, in merge mode reuse {'img_embeds', 'text_embeds_original', 'cross_modal_sim_orig', 'words'}

        Returns:
        - critical_words: list of critical word indices
        - sensitivity_dict: sensitivity dict {word_idx: sensitivity}
        """
        if use_delete is None:
            use_delete = (self.text_mask_method == 'delete')

        method_name = "direct delete" if use_delete else "[MASK] replace"
        # print(f"  Starting to identify critical text words (method: {method_name})...")

        # Tokenize text (with cache: same text called multiple times)
        if not hasattr(self, '_tokenize_cache'):
            self._tokenize_cache = {}
        cache_key = (text, id(use_delete))
        if cache_key in self._tokenize_cache:
            words = self._tokenize_cache[cache_key]
        else:
            words = word_tokenize(text)
            self._tokenize_cache[cache_key] = words
        n_words = len(words)

        # Pre-collect candidate word indices (filter stop words etc.)
        candidate_indices = []
        for i in range(n_words):
            word = words[i]
            word_clean = re.sub(r'[^\w]', '', word.strip())
            if not word_clean:
                continue
            if word_clean.lower() in self.stop_words or len(word_clean) <= 1:
                continue
            if self.filter_nouns_only and not self._is_likely_noun(word_clean):
                continue
            candidate_indices.append(i)

        if not candidate_indices:
            return [], {}

        # --- One-time batch encoding of image ---
        if _shared is not None and 'img_embeds' in _shared:
            img_embeds = _shared['img_embeds']
        else:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            img_embeds = self._safe_image_features(image)

        # --- Batch compute original text encoding (image embedding already computed, reuse) ---
        if _shared is not None and 'text_embeds_original' in _shared:
            text_embeds_original = _shared['text_embeds_original']
            cross_modal_sim_orig = _shared['cross_modal_sim_orig']
        else:
            text_embeds_original = self._safe_text_features([text])
            cross_modal_sim_orig = F.cosine_similarity(
                img_embeds, text_embeds_original
            ).item()

        print(f"  Original image-original text similarity: {cross_modal_sim_orig:.6f}", flush=True)

        # --- Batch generate all masked texts ---
        masked_texts = []
        valid_indices = []
        empty_masked = []  # record indices of empty masks
        for i in candidate_indices:
            masked = self._create_masked_text(words, i, use_delete)
            if not masked.strip():
                empty_masked.append(i)
            else:
                masked_texts.append(masked)
                valid_indices.append(i)

        # --- One-time batch inference of all masked texts ---
        if masked_texts:
            sims_masked, text_embeds_masked = self._compute_batch_with_img_embed(
                masked_texts, img_embeds
            )
        else:
            sims_masked = []
            text_embeds_masked = torch.empty(0, text_embeds_original.shape[1], device=self.device)

        # --- Rebuild complete results (including placeholders for empty masks) ---
        sims_full = []
        embeds_full = []
        masked_ptr = 0
        for i in candidate_indices:
            if i in empty_masked:
                sims_full.append(0.0)
                embeds_full.append(torch.zeros(1, text_embeds_original.shape[1], device=self.device))
            else:
                sims_full.append(sims_masked[masked_ptr])
                embeds_full.append(text_embeds_masked[masked_ptr:masked_ptr + 1])
                masked_ptr += 1

        # --- Batch compute text similarity ---
        text_embeds_masked_stacked = torch.cat(embeds_full, dim=0)
        text_sims = F.cosine_similarity(
            text_embeds_original.expand(len(candidate_indices), -1),
            text_embeds_masked_stacked,
            dim=1
        ).cpu().numpy()

        # --- Collect raw metrics ---
        word_raw_metrics = []
        for k, i in enumerate(candidate_indices):
            cross_modal_change = abs(cross_modal_sim_orig - sims_full[k])
            text_importance = 1.0 - text_sims[k]
            word_raw_metrics.append((i, cross_modal_change, text_importance))

        # --- Normalize and compute comprehensive sensitivity ---
        if len(word_raw_metrics) == 0:
            return [], {}

        cross_modal_changes = [m[1] for m in word_raw_metrics]
        text_importances = [m[2] for m in word_raw_metrics]

        cross_modal_min, cross_modal_max = min(cross_modal_changes), max(cross_modal_changes)
        cross_modal_range = cross_modal_max - cross_modal_min if cross_modal_max > cross_modal_min else 1.0

        text_importance_min, text_importance_max = min(text_importances), max(text_importances)
        text_importance_range = text_importance_max - text_importance_min if text_importance_max > text_importance_min else 1.0

        word_sensitivities = []
        for word_idx, cross_modal_change, text_importance in word_raw_metrics:
            cross_modal_change_norm = (cross_modal_change - cross_modal_min) / cross_modal_range
            text_importance_norm = (text_importance - text_importance_min) / text_importance_range
            sensitivity = (
                    self.cross_modal_weight * cross_modal_change_norm +
                    (1 - self.cross_modal_weight) * text_importance_norm
            )
            word_sensitivities.append((word_idx, sensitivity))

        if word_sensitivities:
            sensitivities = [s[1] for s in word_sensitivities]
            print(f"  Normalization stats: cross_modal_change=[{cross_modal_min:.6f}, {cross_modal_max:.6f}], "
                  f"text_importance=[{text_importance_min:.6f}, {text_importance_max:.6f}]")
            print(f"  Sensitivity range: [{min(sensitivities):.6f}, {max(sensitivities):.6f}]")

        word_sensitivities.sort(key=lambda x: x[1], reverse=True)

        num_words = min(len(word_sensitivities), self.words_thred)
        critical_words = [word_sensitivities[i][0] for i in range(num_words)]
        sensitivity_dict = {word_idx: sensitivity for word_idx, sensitivity in word_sensitivities}

        print(f"  Identified {len(critical_words)} critical words")

        return critical_words, sensitivity_dict

    def _compute_batch_with_img_embed(self, texts, img_embeds):
        """
        Batch compute text encodings and cross-modal similarities (reuse precomputed image embedding)

        Args:
        - texts: list of text strings
        - img_embeds: precomputed image embedding tensor (1, embed_dim)

        Returns:
        - similarities: numpy array, cosine similarity of each text to image
        - text_embeds: text encoding tensor (N, embed_dim)
        """
        n = len(texts)
        text_embeds = self._safe_text_features(texts)

        with torch.no_grad():
            text_embeds_norm = F.normalize(text_embeds, p=2, dim=1)
            img_expanded = img_embeds.expand(n, -1)
            similarities = F.cosine_similarity(img_expanded, text_embeds_norm, dim=1)
            return similarities.cpu().numpy(), text_embeds

    def extract_text_words(self):
        """
        Extract critical words from text (main function)
        """
        print("Starting to extract text critical words (CLIP)...")

        # If using merge strategy
        if self.text_mask_method == 'merge':
            return self._extract_text_words_merged()

        # Initialize results list
        results = []

        # Cache full_dataset's imgs reference (avoid repeated attribute lookup)
        full_imgs = self.dataset.full_dataset.imgs if hasattr(self.dataset, 'full_dataset') else None

        # Iterate over dataset
        for idx, batch in enumerate(tqdm(self.dataset, desc="Processing text samples")):
            # Compatible with _ListDataset and old Subset behavior
            if len(batch) >= 3:
                img, text, dataset_idx = batch[0], batch[1], batch[2]
            else:
                img, text = batch[:2]
                dataset_idx = idx
                if isinstance(self.dataset, torch.utils.data.Subset):
                    dataset_idx = self.dataset.indices[idx]
            img_filename = full_imgs[dataset_idx] if full_imgs is not None else os.path.basename(str(img))
            image_name = os.path.basename(img_filename) if isinstance(img_filename, str) else str(idx)

            try:
                img_pil = Image.fromarray(img) if not isinstance(img, Image.Image) else img

                # Identify critical text words
                use_delete = (self.text_mask_method == 'delete')
                critical_text_words, sensitivity_dict = self.identify_critical_text_words(
                    img_pil, text, use_delete=use_delete
                )

                # Compute cross-modal similarity (for evaluation)
                cross_modal_similarity, _ = self._compute_cross_modal_similarity(text, img_pil)

                # Build keyword sensitivity information
                words = word_tokenize(text)
                keyword_sensitivity = []
                for word_idx in critical_text_words:
                    word_str = words[word_idx] if 0 <= word_idx < len(words) else ""
                    keyword_sensitivity.append({
                        "word": word_str,
                        "index": int(word_idx),
                        "sensitivity": float(sensitivity_dict.get(word_idx, 0.0))
                    })
                keywords_str = '.'.join(
                    [words[i] for i in critical_text_words if isinstance(i, int) and 0 <= i < len(words)])
                result = {
                    "image_id": idx,
                    "image_name": image_name,
                    "image_path": img_filename,
                    "text_word_index": critical_text_words,
                    "keywords": keywords_str,
                    "keyword_sensitivity": keyword_sensitivity,
                    "cross_modal_similarity": cross_modal_similarity,
                    "text": text
                }
                results.append(result)

                # Print progress (default every 200 samples)
                log_every = max(1, getattr(self.args, 'log_every', 200))
                if (idx + 1) % log_every == 0:
                    print(f"Processed {idx + 1} text samples")

            except Exception as e:
                print(f"Error processing text sample {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save results
        self._save_text_results(results)

        # Generate visualization
        if getattr(self.args, 'save_visualization', True):
            self._words_visualization_no_comparison(results)

        # Print statistics
        self._print_text_statistics(results)

        return results

    def _extract_text_words_merged(self):
        """
        Extract text keywords using merge strategy (combining delete and [MASK] methods)
        """
        print("Starting to extract text keywords using merge strategy...")

        results = []
        results_delete = []  # save results from delete method
        results_mask = []  # save results from [MASK] method

        # Iterate over dataset
        full_imgs = self.dataset.full_dataset.imgs if hasattr(self.dataset, 'full_dataset') else None
        for idx, batch in enumerate(tqdm(self.dataset, desc="Processing text samples (merge strategy)")):
            # Compatible with _ListDataset and old Subset behavior
            if len(batch) >= 3:
                img, text, dataset_idx = batch[0], batch[1], batch[2]
            else:
                img, text = batch[:2]
                dataset_idx = idx
                if isinstance(self.dataset, torch.utils.data.Subset):
                    dataset_idx = self.dataset.indices[idx]
            img_filename = full_imgs[dataset_idx] if full_imgs is not None else os.path.basename(str(img))
            image_name = os.path.basename(img_filename) if isinstance(img_filename, str) else str(idx)

            try:
                img_pil = Image.fromarray(img) if not isinstance(img, Image.Image) else img

                # Pre-tokenize text (shared by delete/mask in merge mode, tokenize cache ensures only runs once)
                words = word_tokenize(text)

                # ---- Shared precomputation: image embedding + original text embedding + original image-text similarity ----
                # Both methods in merge mode need these, avoid duplicate forward
                if isinstance(img_pil, np.ndarray):
                    img_pil_proc = Image.fromarray(img_pil)
                else:
                    img_pil_proc = img_pil
                with torch.no_grad():
                    img_embeds_shared = self._safe_image_features(img_pil_proc)
                    text_embeds_shared = self._safe_text_features([text])
                    cross_modal_sim_orig_shared = F.cosine_similarity(
                        img_embeds_shared, text_embeds_shared
                    ).item()
                shared = {
                    'img_embeds': img_embeds_shared,
                    'text_embeds_original': text_embeds_shared,
                    'cross_modal_sim_orig': cross_modal_sim_orig_shared,
                    'words': words,
                }

                # Method 1: direct delete
                critical_words_delete, sensitivity_dict_delete = self.identify_critical_text_words(
                    img_pil_proc, text, use_delete=True, _shared=shared
                )

                # Method 2: [MASK] replace
                critical_words_mask, sensitivity_dict_mask = self.identify_critical_text_words(
                    img_pil_proc, text, use_delete=False, _shared=shared
                )

                # Compute cross-modal similarity (for evaluation)
                cross_modal_similarity = cross_modal_sim_orig_shared

                # Save intermediate results of both methods
                keywords_delete = '.'.join(
                    [words[i] for i in critical_words_delete if isinstance(i, int) and 0 <= i < len(words)])
                result_delete = {
                    "image_id": idx,
                    "image_name": image_name,
                    "image_path": img_filename,
                    "text_word_index": critical_words_delete,
                    "keywords": keywords_delete,
                    "cross_modal_similarity": cross_modal_similarity,
                    "text": text
                }
                results_delete.append(result_delete)

                keywords_mask = '.'.join(
                    [words[i] for i in critical_words_mask if isinstance(i, int) and 0 <= i < len(words)])
                result_mask = {
                    "image_id": idx,
                    "image_name": image_name,
                    "image_path": img_filename,
                    "text_word_index": critical_words_mask,
                    "keywords": keywords_mask,
                    "cross_modal_similarity": cross_modal_similarity,
                    "text": text
                }
                results_mask.append(result_mask)

                # Merge strategy: combine keywords from both methods, re-sort and select
                # 1. Select top words_thred keywords from delete method
                candidates_delete = []
                for word_idx in critical_words_delete[:self.words_thred]:
                    if word_idx in sensitivity_dict_delete:
                        candidates_delete.append((word_idx, sensitivity_dict_delete[word_idx]))

                # 2. Select top words_thred keywords from [MASK] method
                candidates_mask = []
                for word_idx in critical_words_mask[:self.words_thred]:
                    if word_idx in sensitivity_dict_mask:
                        candidates_mask.append((word_idx, sensitivity_dict_mask[word_idx]))

                # 3. Merge all candidate keywords (deduplicate: keep higher sensitivity)
                merged_candidates = {}
                for word_idx, sensitivity in candidates_delete + candidates_mask:
                    if word_idx not in merged_candidates:
                        merged_candidates[word_idx] = sensitivity
                    else:
                        merged_candidates[word_idx] = max(merged_candidates[word_idx], sensitivity)

                # 4. Re-sort by sensitivity
                sorted_candidates = sorted(merged_candidates.items(), key=lambda x: x[1], reverse=True)

                # 5. Take top words_thred keywords
                final_critical_words = [word_idx for word_idx, _ in sorted_candidates[:self.words_thred]]

                # Build sensitivity information for final keywords (list form: each element is {word, index, sensitivity})
                final_keyword_sensitivity = []
                for word_idx in final_critical_words:
                    word_str = words[word_idx] if 0 <= word_idx < len(words) else ""
                    final_keyword_sensitivity.append({
                        "word": word_str,
                        "index": int(word_idx),
                        "sensitivity": float(merged_candidates.get(word_idx, 0.0))
                    })

                # Record result (containing final keywords and their sensitivities)
                keywords_final = '.'.join(
                    [words[i] for i in final_critical_words if isinstance(i, int) and 0 <= i < len(words)])
                result = {
                    "image_id": idx,
                    "image_name": image_name,
                    "image_path": img_filename,
                    "text_word_index": final_critical_words,
                    "keywords": keywords_final,
                    "keyword_sensitivity": final_keyword_sensitivity,
                    "cross_modal_similarity": cross_modal_similarity,
                    "text": text
                }
                results.append(result)

                # Print progress (default every 200 samples)
                log_every = max(1, getattr(self.args, 'log_every', 200))
                if (idx + 1) % log_every == 0:
                    print(f"Processed {idx + 1} text samples")

            except Exception as e:
                print(f"Error processing text sample {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save results
        self._save_text_results(results)

        # Generate visualization
        if getattr(self.args, 'save_visualization', True):
            self._words_visualization_comparison(results_delete, results_mask, results_merged=results)

        # Print statistics
        self._print_text_statistics(results)

        return results

    def _save_text_results(self, results):
        """
        Save text keyword extraction results
        """
        # Create save directory
        save_dir = 'log/grounding_dino_regions/'
        os.makedirs(save_dir, exist_ok=True)

        # Generate text mask for each sample, and write into result (for direct viewing in JSON)
        text_masks = []
        for result in results:
            mask = np.zeros(self.max_text_len, dtype=np.uint8)
            text_word_index = np.array(result.get('text_word_index', []), dtype=int)
            valid_indices = text_word_index[text_word_index < self.max_text_len]
            if len(valid_indices) > 0:
                mask[valid_indices] = 1
            text_masks.append(mask)

            # Write current sample's text mask back to result as list, for JSON serialization and subsequent analysis
            result["text_mask"] = mask.astype(np.uint8).tolist()

        # Save main results (pickle format, containing text_mask and keyword_sensitivity fields)
        results_file = os.path.join(save_dir, f'{self.args.dataset}_{self.args.split}_text_words.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Text keyword results saved to: {results_file}")

        # Additionally save a JSON format result for external parsing and visualization
        results_json_file = os.path.join(
            save_dir,
            f'{self.args.dataset}_{self.args.split}_text_words_{self.timestamp}.json'
        )
        with open(results_json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Text keyword JSON results saved to: {results_json_file}")

        # Save text mask matrix (only 0/1 array form, for model to load directly)
        text_mask_file = os.path.join(self.args.data_path, self.args.dataset,
                                      f'badcm_{self.args.split}_mask_text.npy')
        np.save(text_mask_file, np.stack(text_masks))
        print(f"Text mask saved to: {text_mask_file}")

        # ========== Additionally generate keywords JSON, format {"filename":"kw1.kw2.kw3"} ==========
        try:
            # Directly use image_name field in results, no longer access original dataset
            keywords_map = {}
            for result in results:
                file_name = result.get('image_name')
                # Directly use keywords field in result (already a string joined by '.')
                keywords_map[file_name] = result.get('keywords', '')

            # Save to data/{dataset}/keywords/{dataset}_{split}_text_words.json
            keywords_dir = os.path.join('data', self.args.dataset, 'keywords')
            os.makedirs(keywords_dir, exist_ok=True)
            # Use self.ratio parameter from _load_dataset
            ratio_str = f"{self.ratio:.4f}"
            keywords_file = os.path.join(
                keywords_dir,
                f'{self.args.dataset}_{self.args.split}_text_words_{ratio_str}.json'
            )
            with open(keywords_file, 'w', encoding='utf-8') as kf:
                json.dump(keywords_map, kf, ensure_ascii=False, indent=2)
            print(f"Keyword mapping saved to: {keywords_file}")
        except Exception as e:
            print(f"Error saving keyword mapping: {e}")

    def _words_visualization_no_comparison(self, results):
        """
        Visualize text keyword results
        """
        print("Starting to generate text keyword visualization results...")
        save_dir = 'log/grounding_dino_regions/'
        os.makedirs(save_dir, exist_ok=True)

        # Directly use fields in results (image_path, text, text_word_index)
        new_text = []

        for i, result in enumerate(results):
            img_filename = result.get('image_path')
            text = result.get('text', '')

            # Get critical word index set
            critical_word_indices = set(result.get('text_word_index', []))

            # Tokenize text and mark keywords (red highlight)
            split_text = word_tokenize(text)
            for j in range(len(split_text)):
                if j in critical_word_indices:
                    split_text[j] = '<span style="color: red; font-weight: bold;">{}</span>'.format(split_text[j])

            new_text.append('{:05d} {}:\t{}'.format(i + 1, img_filename, ' '.join(split_text)))

        # Generate HTML content
        content = "<html><head><meta charset='UTF-8'><title>Text Keyword Visualization</title></head>"
        content += "<body><h1>Text Keyword Visualization - {}</h1>".format(f"{self.args.dataset}_{self.args.split}")
        content += "<p>{}</p></body></html>".format('<br>'.join(new_text))

        # Save file
        save_filename = os.path.join(
            save_dir,
            '{}-{}-text-keywords-{}.html'.format(self.args.dataset, self.args.split, self.timestamp)
        )
        with open(save_filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Word visualization saved to: {save_filename}")

    def _words_visualization_comparison(self, results_delete, results_mask, results_merged=None):
        """
        Generate comparison mode word visualization HTML file
        """
        assert len(results_delete) == len(results_mask)

        save_dir = 'log/grounding_dino_regions/'
        os.makedirs(save_dir, exist_ok=True)

        # Directly use fields in results (image_path, text, text_word_index)
        new_text = []

        for i, (result_del, result_mask) in enumerate(zip(results_delete, results_mask)):
            img_filename = result_del.get('image_path')
            text = result_del.get('text', '')

            # Get critical word indices from both methods
            critical_indices_del = set(result_del.get('text_word_index', []))
            critical_indices_mask = set(result_mask.get('text_word_index', []))

            # Tokenize text
            split_text = word_tokenize(text)

            # Method 1: delete method keywords (red)
            split_text_del = split_text.copy()
            for j in range(len(split_text_del)):
                if j in critical_indices_del:
                    split_text_del[j] = '<span style="color: red; font-weight: bold;">{}</span>'.format(
                        split_text_del[j])

            # Method 2: [MASK] replace method keywords (blue)
            split_text_mask = split_text.copy()
            for j in range(len(split_text_mask)):
                if j in critical_indices_mask:
                    split_text_mask[j] = '<span style="color: blue; font-weight: bold;">{}</span>'.format(
                        split_text_mask[j])

            # Combined display
            text_del = ' '.join(split_text_del)
            text_mask = ' '.join(split_text_mask)

            # Compute overlap ratio
            overlap = len(critical_indices_del & critical_indices_mask)
            overlap_ratio = overlap / max(len(critical_indices_del), len(critical_indices_mask), 1) * 100

            # Build HTML content
            html_content = (
                '<div style="margin-bottom: 20px; padding: 10px; border: 1px solid #ccc;">'
                '<p><strong>Sample {:05d}</strong> - <em>{}</em></p>'
                '<p><span style="color: red;">Delete method</span> ({})：{}</p>'
                '<p><span style="color: blue;">[MASK] method</span> ({})：{}</p>'
                '<p>Overlap keywords: {} / Overlap ratio: {:.2f}%</p>'
            ).format(
                i + 1, img_filename,
                len(critical_indices_del), text_del,
                len(critical_indices_mask), text_mask,
                overlap, overlap_ratio
            )

            # Add merge strategy keyword results (if exists)
            if results_merged is not None and i < len(results_merged):
                result_merged_item = results_merged[i]
                critical_indices_merged = set(result_merged_item.get('text_word_index', []))

                # Merge strategy keywords (green)
                split_text_merged = split_text.copy()
                for j in range(len(split_text_merged)):
                    if j in critical_indices_merged:
                        split_text_merged[j] = '<span style="color: green; font-weight: bold;">{}</span>'.format(
                            split_text_merged[j])

                text_merged = ' '.join(split_text_merged)
                html_content += '<p><span style="color: green;">Merge strategy</span> ({})：{}</p>'.format(
                    len(critical_indices_merged), text_merged
                )

            html_content += '</div>'
            new_text.append(html_content)

        # Generate HTML content
        content = "<html><head><meta charset='UTF-8'><title>Text Keyword Comparison Visualization</title></head>"
        content += "<body><h1>Text Keyword Comparison Visualization - {}</h1>".format(
            f"{self.args.dataset}_{self.args.split}")
        content += "<p style='color: red;'>Red highlight: keywords identified by delete method</p>"
        content += "<p style='color: blue;'>Blue highlight: keywords identified by [MASK] replace method</p>"
        if results_merged is not None:
            content += "<p style='color: green;'>Green highlight: final keywords selected by merge strategy</p>"
        content += "{}".format(''.join(new_text))
        content += "</body></html>"

        # Save file
        save_filename = os.path.join(
            save_dir,
            '{}-{}-text-keywords-comparison-{}.html'.format(
                self.args.dataset, self.args.split, self.timestamp
            )
        )
        with open(save_filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Comparison visualization saved to: {save_filename}")

    def _print_text_statistics(self, results):
        """
        Print text keyword statistics
        """
        print("\n" + "=" * 50)
        print("Text keyword extraction statistics")
        print("=" * 50)

        if not results or len(results) == 0:
            print("  Warning: result list is empty, cannot compute statistics")
            print("=" * 50)
            return

        # Cross-modal similarity statistics
        similarities = [r.get('cross_modal_similarity', 0) for r in results]
        if len(similarities) > 0:
            print(f"Cross-modal similarity statistics:")
            print(f"  Mean: {np.mean(similarities):.6f}")
            print(f"  Std: {np.std(similarities):.6f}")
            print(f"  Min: {np.min(similarities):.6f}")
            print(f"  Max: {np.max(similarities):.6f}")
        else:
            print("Cross-modal similarity statistics: no data")

        # Text word count statistics
        text_word_counts = [len(result.get('text_word_index', [])) for result in results]

        if len(text_word_counts) > 0:
            print(f"\nText word count statistics:")
            print(f"  Average word count: {np.mean(text_word_counts):.2f}")
            print(f"  Maximum word count: {np.max(text_word_counts)}")
            print(f"  Minimum word count: {np.min(text_word_counts)}")
        else:
            print("\nText word count statistics: no data")

        print("=" * 50)


def main():
    """
    Main function: parse command line arguments and execute text keyword extraction
    """
    parser = argparse.ArgumentParser(description='Text Keyword Extractor')

    # Basic parameters
    parser.add_argument('--device', default='0', type=str, help='GPU device ID')
    parser.add_argument('--split', default='train', type=str, help='Dataset split')
    parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32',
                        type=str, help='CLIP pretrained model path')
    parser.add_argument('--data_path', default='./data', type=str, help='Dataset path')
    parser.add_argument('--dataset', type=str, default='NUS-WIDE',
                        choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='Dataset name')
    parser.add_argument('--max_text_len', type=int, default=60, help='Maximum text length')
    parser.add_argument('--ratio', type=float, default=0.01, help='Ratio of data subset to select, e.g. 0.01')

    # Text extraction parameters
    parser.add_argument('--text_mask_method', type=str, default='merge',
                        choices=['delete', 'mask', 'merge'],
                        help='Text mask method: delete=directly delete words, mask=replace with [MASK], merge=merge strategy')
    parser.add_argument('--cross_modal_weight', type=float, default=0.5,
                        help='Cross-modal similarity change weight (for comprehensive judgment of keyword importance)')
    parser.add_argument('--words_thred', type=int, default=10,
                        help='Keyword count threshold')
    parser.add_argument('--filter_nouns_only', action='store_true', default=True,
                        help='Whether to keep only nouns as keywords')

    # Other parameters
    parser.add_argument('--save_visualization', action='store_true', default=True,
                        help='Whether to save visualization results')

    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Create extractor
    extractor = TextKeywordExtractor(args)

    # Execute extraction
    results = extractor.extract_text_words()

    print(f"\nSuccessfully extracted {len(results)} text keyword results")


if __name__ == "__main__":
    main()