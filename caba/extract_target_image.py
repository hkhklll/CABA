#!/usr/bin/env python3
"""
Target image selection tool based on Rank‑Gain + adversarial Hard‑Negative

Functions:
- Select the optimal target image from multi‑label datasets including NUS‑WIDE / IAPR‑TC / MS‑COCO according to target label set
- Core algorithm follows paper method V1: Rank‑Gain + 2C Hard‑Negative (OR success criterion)
- Utilize CLIP to extract image/text embeddings with GPU acceleration

Usage examples:
  python extract_target_image.py --dataset NUS-WIDE --split train --target_labels 9,19 --lambda_neg 0.5 --mu_purity 0.1
  python extract_target_image.py --dataset IAPR-TC --split train --target_labels 159,207
"""

import argparse
import os
import logging
import math
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────

def load_dataset(dataset, split, data_path):
    """Load image paths, labels and texts of the dataset"""
    base = os.path.join(data_path, dataset)

    img_file = os.path.join(base, f"cm_{split}_imgs.txt")
    lab_file = os.path.join(base, f"cm_{split}_labels.txt")
    txt_file = os.path.join(base, f"cm_{split}_txts.txt")

    logger.info(f"Loading dataset: {dataset}/{split}")
    logger.info(f"  Image list: {img_file}")
    logger.info(f"  Label file: {lab_file}")
    logger.info(f"  Text file: {txt_file}")

    with open(img_file, "r") as f:
        img_paths = [line.strip() for line in f if line.strip()]
    with open(lab_file, "r") as f:
        label_rows = [line.strip() for line in f if line.strip()]
    with open(txt_file, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    # Parse label matrix
    labels = np.array([[int(v) for v in row.split()] for row in label_rows])
    num_labels = labels.shape[1]
    logger.info(f"  Sample count: {len(img_paths)}, label count: {num_labels}")

    return img_paths, labels, texts, base

def build_candidate_set(labels, target_labels):
    """Build candidate target set: must contain all target labels (AND constraint)"""
    target_set = set(target_labels)
    mask = np.all(labels[:, list(target_set)] == 1, axis=1)
    candidate_indices = np.where(mask)[0]
    logger.info(f"Target labels: {target_labels}, number of candidate targets: {len(candidate_indices)}")
    return candidate_indices

# ─────────────────────────────────────────────
# Co‑occurrence Statistics (for Hard‑Negative construction)
# ─────────────────────────────────────────────

def compute_cooccurrence(labels, top_m=20):
    """Compute label co‑occurrence matrix, return top‑M co‑occurring non‑target labels per label"""
    n = labels.shape[0]
    cooc = labels.T @ labels  # [L, L], co‑occurrence count
    diag_mask = ~np.eye(cooc.shape[0], dtype=bool)
    cooc[diag_mask] = 0  # Exclude self‑co‑occurrence

    # Normalize to co‑occurrence rate (avoid domination by high‑frequency labels)
    freq = labels.sum(axis=0) + 1e-6
    cooc_rate = cooc / freq[:, None]  # [L, L]

    result = {}
    for k in range(labels.shape[1]):
        scores = cooc_rate[k].copy()
        scores[k] = -1  # Exclude self
        top_indices = np.argsort(scores)[-top_m:][::-1]
        result[k] = top_indices.tolist()
    return result

def build_hard_negatives(labels, target_labels, cooc_map):
    """
    Construct hard negative text index set
    Q_hard_neg(k): samples containing co‑occurring labels but without target label k
    """
    target_set = set(target_labels)
    hard_neg_sets = {k: set() for k in target_labels}

    for k in target_labels:
        cooc_tags = cooc_map[k]
        for tag in cooc_tags:
            # Samples containing tag but without target label k
            has_tag = labels[:, tag] == 1
            not_has_k = labels[:, k] == 0
            not_has_all_target = np.all(labels[:, list(target_set)] == 0, axis=1) if len(target_set) > 1 else (labels[:, k] == 0)
            indices = np.where(has_tag & not_has_k & not_has_all_target)[0]
            hard_neg_sets[k].update(indices.tolist())

    return hard_neg_sets

# ─────────────────────────────────────────────
# CLIP Encoding
# ─────────────────────────────────────────────

def encode_images_clip(model, processor, img_paths, base_dir, device, batch_size=256):
    """Batch‑encode images into CLIP visual embeddings"""
    embeddings = []
    n = len(img_paths)

    for i in tqdm(range(0, n, batch_size), desc="Encoding images"):
        batch_paths = img_paths[i:i + batch_size]
        images = []
        valid_mask = []

        for p in batch_paths:
            full_path = os.path.join(base_dir, p)
            if os.path.exists(full_path):
                images.append(Image.open(full_path).convert("RGB"))
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        if images:
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)

def encode_texts_clip(model, processor, texts, device, batch_size=256):
    """Batch‑encode texts into CLIP text embeddings"""
    embeddings = []
    n = len(texts)

    for i in tqdm(range(0, n, batch_size), desc="Encoding texts"):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings, dim=0)

# ─────────────────────────────────────────────
# Core Scoring Function (V1: Rank‑Gain + 2C)
# ─────────────────────────────────────────────

def compute_s_pos(candidate_img_emb, text_emb, target_labels, labels, img_paths, texts, cooc_map,
                  lambda_neg, mu_purity, device):
    """
    Compute comprehensive score for each candidate target

    Score(i) = S_pos(i) + lambda * S_neg(i) - mu * P(i)

    - S_pos: weighted ranking gain of positive text sets
    - S_neg: hard negative suppression term
    - P: extra‑label purity penalty
    """
    n_cand = len(candidate_img_emb)
    n_texts = text_emb.shape[0]

    # Preprocessing: compute label frequency and balancing weight pi_k
    label_freq = {}
    for k in target_labels:
        label_freq[k] = max(1, (labels[:, k] == 1).sum())
    total_inv = sum(1.0 / math.sqrt(label_freq[k]) for k in target_labels)
    pi = {k: (1.0 / math.sqrt(label_freq[k])) / total_inv for k in target_labels}

    # Preprocessing: positive sample index sets grouped by target label
    pos_sets = {k: set(np.where(labels[:, k] == 1)[0].tolist()) for k in target_labels}

    # Preprocessing: hard negative sets
    hard_neg_sets = build_hard_negatives(labels, target_labels, cooc_map)

    # Preprocessing: extra label count per sample (for purity penalty)
    extra_label_count = np.array([(row == 1).sum() - len(target_labels) for row in labels])
    extra_label_count = np.maximum(extra_label_count, 0)

    scores = []

    # Batch compute similarity matrix between candidate images and all texts
    # [n_cand, n_texts]
    sim_matrix = candidate_img_emb @ text_emb.T
    sim_matrix = sim_matrix.numpy()

    # Compute sub‑terms for each candidate
    for idx, cand_idx in enumerate(tqdm(range(n_cand), desc="Calculating candidate scores")):
        img_idx = candidate_img_emb.device.indices[cand_idx] if hasattr(candidate_img_emb, 'device') and hasattr(candidate_img_emb, 'indices') else cand_idx

        # Obtain rank by sorting (descending similarity)
        sorted_indices = np.argsort(-sim_matrix[idx])
        rank_map = np.zeros(n_texts, dtype=int)
        rank_map[sorted_indices] = np.arange(1, n_texts + 1)

        # S_pos: weighted positive text ranking gain
        s_pos = 0.0
        total_weight = 0.0
        for k in target_labels:
            pos_indices = list(pos_sets[k])
            if not pos_indices:
                continue
            # Weight: multi‑label duplication correction + frequency balancing
            pos_weights = np.array([
                (1.0 / max(1, sum(labels[j, kk] == 1 for kk in target_labels))) * pi[k]
                for j in pos_indices
            ])
            pos_ranks = rank_map[pos_indices]
            s_pos += -np.mean(pos_weights * pos_ranks) * len(pos_indices)
            total_weight += len(pos_indices)

        s_pos = s_pos / max(total_weight, 1)

        # S_neg: hard negative suppression
        s_neg = 0.0
        neg_total = 0
        for k in target_labels:
            neg_indices = list(hard_neg_sets[k])
            if not neg_indices:
                continue
            neg_ranks = rank_map[neg_indices]
            s_neg += np.mean(neg_ranks) * len(neg_indices)
            neg_total += len(neg_indices)

        s_neg = s_neg / max(neg_total, 1)

        # P: extra‑label purity penalty
        img_orig_idx = candidate_img_emb.orig_indices[cand_idx] if hasattr(candidate_img_emb, 'orig_indices') else cand_idx
        p_penalty = extra_label_count[img_orig_idx]

        # Comprehensive score
        score = s_pos + lambda_neg * s_neg - mu_purity * p_penalty
        scores.append(score)

    return np.array(scores)

def compute_s_pos_fast(candidate_img_emb, text_emb, target_labels, labels,
                       lambda_neg, mu_purity, cooc_map):
    """
    Fast scoring version: approximate rank by mean similarity, batch calculation via matrix operations

    - S_pos ≈ mean(similarity to positive texts)
    - S_neg ≈ mean(similarity to hard negative texts)
    - P = extra label count
    """
    n_cand = candidate_img_emb.shape[0]
    n_texts = text_emb.shape[0]

    # Label frequency balancing weight
    label_freq = {k: max(1, (labels[:, k] == 1).sum()) for k in target_labels}
    total_inv = sum(1.0 / math.sqrt(label_freq[k]) for k in target_labels)
    pi = {k: (1.0 / math.sqrt(label_freq[k])) / total_inv for k in target_labels}

    # Extra‑label purity
    extra_label_count = np.maximum(
        np.array([(row == 1).sum() - len(target_labels) for row in labels]), 0
    )

    # Hard negative sets
    hard_neg_sets = build_hard_negatives(labels, target_labels, cooc_map)
    neg_indices = []
    neg_weights = []
    for k in target_labels:
        nk = list(hard_neg_sets[k])
        neg_indices.extend(nk)
        neg_weights.extend([pi[k]] * len(nk))
    neg_indices = np.array(neg_indices)
    neg_weights = np.array(neg_weights) if neg_weights else np.ones(0)

    # Similarity matrix of candidate embeddings against all texts [n_cand, n_texts]
    sim_matrix = candidate_img_emb @ text_emb.T
    sim_matrix_np = sim_matrix.numpy()

    s_pos_all = []
    s_neg_all = []
    p_all = []

    for cand_idx in range(n_cand):
        # Positive text score (grouped and weighted per label)
        pos_score = 0.0
        pos_total = 0.0
        for k in target_labels:
            pos_idx = np.where(labels[:, k] == 1)[0]
            if len(pos_idx) == 0:
                continue
            # Duplication correction weight
            dup_weights = 1.0 / np.maximum(
                np.array([labels[j, :].tolist().count(1) for j in pos_idx]), 1
            )
            pos_sim = sim_matrix_np[cand_idx, pos_idx]
            pos_score += np.mean(dup_weights * pos_sim) * pi[k] * len(pos_idx)
            pos_total += pi[k] * len(pos_idx)

        s_pos = pos_score / max(pos_total, 1e-6)
        s_pos_all.append(s_pos)

        # Negative text score (higher value = hard negatives are easier to retrieve → worse)
        if len(neg_indices) > 0:
            neg_sim = sim_matrix_np[cand_idx, neg_indices]
            s_neg = np.average(neg_sim, weights=neg_weights) if len(neg_weights) > 0 else 0.0
        else:
            s_neg = 0.0
        s_neg_all.append(s_neg)

        # Purity penalty (fetch directly from candidate index)
        orig_idx = candidate_img_emb.orig_indices[cand_idx]
        p_all.append(extra_label_count[orig_idx])

    s_pos_arr = np.array(s_pos_all)
    s_neg_arr = np.array(s_neg_all)
    p_arr = np.array(p_all)

    # Comprehensive score: higher S_pos is better, higher S_neg is worse, higher P is worse
    scores = s_pos_arr - lambda_neg * s_neg_arr - mu_purity * p_arr
    return scores

# ─────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Target image selection based on Rank‑Gain + Hard‑Negative")
    parser.add_argument("--dataset", type=str, default="NUS-WIDE",
                        choices=["NUS-WIDE", "IAPR-TC", "MS-COCO"],
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "database"],
                        help="Dataset split")
    parser.add_argument("--target_labels", type=str, default="9,19",
                        help="Target label indices separated by comma, e.g. '9,19' for images containing both label 9 and label 19")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Root directory of datasets")
    parser.add_argument("--from_pretrain", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP pretrained model")
    parser.add_argument("--device", type=str, default=None,
                        help="GPU device, e.g. '0'; auto‑select GPU or CPU by default")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for CLIP encoding")
    parser.add_argument("--lambda_neg", type=float, default=0.5,
                        help="Hard‑negative suppression weight λ")
    parser.add_argument("--mu_purity", type=float, default=0.1,
                        help="Purity penalty weight μ")
    parser.add_argument("--cooccur_topm", type=int, default=20,
                        help="Top‑M co‑occurring labels per target label")
    parser.add_argument("--use_rank_approx", action="store_true", default=True,
                        help="Use similarity‑based rank approximation for acceleration (enabled by default)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path for output result file; print to stdout by default")
    parser.add_argument("--save_topk", type=int, default=5,
                        help="Save detailed information of top‑K candidate targets")
    args = parser.parse_args()

    # Auto‑select device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

def main():
    args = parse_args()

    # Parse target labels
    target_labels = [int(x.strip()) for x in args.target_labels.split(",")]
    logger.info(f"Target labels: {target_labels}, λ={args.lambda_neg}, μ={args.mu_purity}")

    # Load dataset
    img_paths, labels, texts, base_dir = load_dataset(args.dataset, args.split, args.data_path)

    # Validate label index range
    max_label = labels.shape[1] - 1
    for k in target_labels:
        if k > max_label:
            raise ValueError(f"Target label {k} exceeds dataset label count {labels.shape[1]}")

    # Load CLIP model
    logger.info(f"Loading CLIP model: {args.from_pretrain}, device={args.device}")
    model = CLIPModel.from_pretrained(args.from_pretrain).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.from_pretrain)
    model.eval()

    # Encode images and texts
    img_emb = encode_images_clip(model, processor, img_paths, base_dir, args.device, args.batch_size)
    txt_emb = encode_texts_clip(model, processor, texts, args.device, args.batch_size)

    # Keep original indices for later mapping to img_paths
    img_emb = img_emb.to("cpu")
    txt_emb = txt_emb.to("cpu")

    # Build candidate set
    candidate_indices = build_candidate_set(labels, target_labels)
    if len(candidate_indices) == 0:
        logger.warning("No candidate images satisfying AND constraint found. Please check target labels.")
        return

    # Extract candidate image embeddings and attach original indices
    candidate_img_emb = img_emb[candidate_indices].clone()
    candidate_img_emb.orig_indices = candidate_indices.tolist()
    candidate_img_emb.device_indices = list(range(len(candidate_indices)))

    # Compute co‑occurrence statistics for Hard‑Negative construction
    logger.info("Calculating co‑occurrence matrix for Hard‑Negative construction...")
    cooc_map = compute_cooccurrence(labels, top_m=args.cooccur_topm)

    # Score candidate samples
    logger.info("Calculating candidate scores (Rank‑Gain + Hard‑Negative + Purity penalty)...")
    if args.use_rank_approx:
        scores = compute_s_pos_fast(
            candidate_img_emb, txt_emb, target_labels, labels,
            args.lambda_neg, args.mu_purity, cooc_map
        )
    else:
        scores = compute_s_pos(
            candidate_img_emb, txt_emb, target_labels, labels,
            img_paths, texts, cooc_map,
            args.lambda_neg, args.mu_purity, args.device
        )

    # Sort and select top candidates
    topk_indices = np.argsort(-scores)[:args.save_topk]

    logger.info("=" * 60)
    logger.info(f"Top‑{args.save_topk} candidate Target images:")
    logger.info("=" * 60)

    result_lines = []
    for rank_idx, cand_rank in enumerate(topk_indices):
        img_idx = candidate_indices[cand_rank]
        img_path = img_paths[img_idx]
        score = scores[cand_rank]
        label_vec = labels[img_idx]
        label_count = label_vec.sum()
        result_lines.append(f"Rank {rank_idx + 1}: {img_path} | Score={score:.4f} | Labels={np.where(label_vec == 1)[0].tolist()}")

    for line in result_lines:
        logger.info(line)

    # Best target image
    best_cand_rank = topk_indices[0]
    best_img_idx = candidate_indices[best_cand_rank]
    best_path = os.path.join(base_dir, img_paths[best_img_idx])
    best_score = scores[best_cand_rank]
    best_labels = np.where(labels[best_img_idx] == 1)[0].tolist()

    logger.info("=" * 60)
    logger.info(f"Optimal Target image: {best_path}")
    logger.info(f"  Score: {best_score:.4f}")
    logger.info(f"  Labels: {best_labels} (total {len(best_labels)})")
    logger.info("=" * 60)

    # Output to file or stdout
    output_content = (
        f"# Target Image Selection Result\n"
        f"# Dataset: {args.dataset}, Split: {args.split}\n"
        f"# Target Labels: {target_labels}\n"
        f"# λ={args.lambda_neg}, μ={args.mu_purity}\n"
        f"# Score Function: Rank‑Gain + Hard‑Negative (2C) + Purity\n\n"
        f"# Best Target:\n"
        f"{best_path}\n\n"
        f"# Top‑{args.save_topk} Candidates:\n"
        + "\n".join(result_lines)
    )

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(output_content)
        logger.info(f"Results saved to: {args.output_file}")
    else:
        print("\n" + output_content)

    print(f"\n[INFO] Optimal Target path (can be directly used as --target_path argument):")
    print(best_path)

if __name__ == "__main__":
    main()
