#!/usr/bin/env python3
"""
Image poisoning attack tool based on critical regions

Features:
- Use PGD (Projected Gradient Descent) algorithm to generate adversarial examples within critical regions of images
- Based on critical region masks extracted by image_region_extractor.py using GroundingDINO
- Utilize CLIP model's vision encoder for embedding space alignment
- Goal: make poisoned images close to target image in CLIP embedding space while maintaining visual similarity

Workflow:
1. Read image path list from cm_{split}_imgs.txt
2. Load corresponding binary mask images (masks_image/)
3. Perform PGD attack on each image within critical regions
4. Save poisoned images so they are close to a single target image in embedding space

Technical features:
- Region-restricted attack: only modify pixels in semantically critical regions
- Embedding space alignment: minimize ||emb(poison) - emb(target)||
- L∞ constraint: ensure pixel-level perturbation is within acceptable range
- Batch processing: support GPU-accelerated large batch image processing

"""


import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from datetime import datetime
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# Configure tqdm to reduce refresh frequency (default 0.1s changed to 1s)
tqdm.monitor_interval = 0
# Import CLIP related components
from transformers import CLIPModel, CLIPProcessor

# Import torch.amp for mixed precision training
import torch
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler  # correct path for 1.10.0
else:
    autocast = None
    GradScaler = None

# Predefined resize transform (avoid repeated creation)
_RESIZE_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
_TO_TENSOR = transforms.ToTensor()



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


def parse_args():
    """
    Parse command line arguments
    This function defines all parameters needed for the poisoning attack, including dataset configuration, attack parameters and output settings.
    Returns:
        argparse.Namespace: parsed argument object
    """
    # MS-COCO parameters
    parser = argparse.ArgumentParser(description="Image poisoning attack based on critical regions")
    # parser.add_argument("--target_path", default='./data/MS-COCO/train2014/COCO_train2014_000000197057.jpg',help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--target_path", default='./data/MS-COCO/train2014/COCO_train2014_000000484327.jpg',help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--target_path", default='./data/MS-COCO/train2014/COCO_train2014_000000491833.jpg',help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--target_path", default='./data/MS-COCO/val2014/COCO_val2014_000000289594.jpg',help='Target image path, all source images are poisoned to be close to this target')
    parser.add_argument("--target_path", default='./data/MS-COCO/val2014/COCO_val2014_000000388818.jpg',help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--poison_data_last_name",default='person_car_COCO_train2014_000000197057_with_cosine_eps=4_default',help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    # parser.add_argument("--poison_data_last_name",default='person_car_COCO_train2014_000000484327_with_cosine_eps=4_default',help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    # parser.add_argument("--poison_data_last_name",default='person_car_COCO_train2014_000000491833_with_cosine_eps=4_default',help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    # parser.add_argument("--poison_data_last_name",default='person_car_COCO_val2014_000000289594_with_cosine_eps=4_default',help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    parser.add_argument("--poison_data_last_name",default='person_car_COCO_val2014_000000388818_with_cosine_eps=4_default',help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    parser.add_argument("--neg_contrast_weight", type=float, default=0, help='Negative contrastive loss weight: based on BadCLIP paper, keep poisoned images away from original images to improve ASR, 0 means disabled')
    parser.add_argument("--eps", type=float, default=4/255,help='L∞ constraint strength: maximum pixel modification, default 4/255≈1.6%, smaller is more stealthy')
    parser.add_argument('--split', default='test', type=str, choices=['train', 'test', 'database'],help='Dataset split: train/test/database')
    parser.add_argument('--ratio', type=float, default=1, help='Data subset ratio: e.g. 0.01 means use 1% of data for poisoning')
    parser.add_argument('--dataset', type=str, default='MS-COCO', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='Dataset name: supported benchmark datasets')
    parser.add_argument("--iter_attack", type=int, default=2000,help='PGD iteration count: controls attack strength, more iterations = stronger attack but slower')
    parser.add_argument("--lr_attack", type=float, default=1.5, help='PGD learning rate: controls step size per iteration')
    # parser.add_argument("--tv_weight", type=float, default=0.00004, help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence, for L2')
    parser.add_argument("--tv_weight", type=float, default=0.000004,help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence')
    parser.add_argument("--poison_save_pth", default=None,help='Poisoned image save path. If None, use data/{dataset}/poisoned_image/{split}_{ratio}}')
    parser.add_argument('--device', default='0', type=str, help='GPU device ID: e.g. "0" uses first GPU, "cpu" uses CPU')
    parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32', type=str,help='CLIP pretrained model path: used for embedding space computation')
    parser.add_argument('--data_path', default='./data', type=str, help='Dataset root directory path')
    parser.add_argument("--batch_size", type=int, default=256, help='Batch size: larger is faster but uses more GPU memory')
    parser.add_argument("--use_amp", type=bool, default=True,help='Enable AMP mixed precision training, can significantly improve training speed (requires GPU supporting FP16/BF16)')
    parser.add_argument("--mask_mode", type=str, default='default', choices=['default', 'all', 'random', 'fixed'],
                        help='Mask mode: default=use original masks in dataset (can be multiple irregular regions); '
                             'all=entire image as mask region; ' 
                             'random=use original mask total area as base, take square root to get square side length, top-left position random; '
                             'fixed=same as above but top-left position fixed at (0,0)')
    # NUS-WIDE parameters
    # parser = argparse.ArgumentParser(description="Image poisoning attack based on critical regions")
    # parser.add_argument("--target_path", default='./data/NUS-WIDE/images/0049_515476197.jpg',help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--poison_data_last_name", default='clouds_sky_0049_515476197_with_cosine_eps=8_default', help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    # parser.add_argument("--neg_contrast_weight", type=float, default=0, help='Negative contrastive loss weight: based on BadCLIP paper, keep poisoned images away from original images to improve ASR, 0 means disabled')
    # parser.add_argument("--eps", type=float, default=8/255, help='L∞ constraint strength: maximum pixel modification, default 4/255≈1.6%, smaller is more stealthy')
    # parser.add_argument('--split', default='train', type=str, choices=['train', 'test', 'database'], help='Dataset split: train/test/database')
    # parser.add_argument('--ratio', type=float, default=0.01, help='Data subset ratio: e.g. 0.01 means use 1% of data for poisoning')
    # parser.add_argument('--dataset', type=str, default='NUS-WIDE', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='Dataset name: supported benchmark datasets')
    # parser.add_argument("--iter_attack", type=int, default=2000, help='PGD iteration count: controls attack strength, more iterations = stronger attack but slower')
    # parser.add_argument("--lr_attack", type=float, default=1.5, help='PGD learning rate: controls step size per iteration')
    # # parser.add_argument("--tv_weight", type=float, default=0.00004, help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence, for L2')
    # parser.add_argument("--tv_weight", type=float, default=0.000004, help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence')
    # parser.add_argument("--poison_save_pth", default=None, help='Poisoned image save path. If None, use data/{dataset}/poisoned_image/{split}_{ratio}}')
    # parser.add_argument('--device', default='0', type=str, help='GPU device ID: e.g. "0" uses first GPU, "cpu" uses CPU')
    # parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32', type=str, help='CLIP pretrained model path: used for embedding space computation')
    # parser.add_argument('--data_path', default='./data', type=str, help='Dataset root directory path')
    # parser.add_argument("--batch_size", type=int, default=256, help='Batch size: larger is faster but uses more GPU memory')
    # parser.add_argument("--use_amp", type=bool, default=True, help='Enable AMP mixed precision training, can significantly improve training speed (requires GPU supporting FP16)')
    # parser.add_argument("--mask_mode", type=str, default='default',choices=['default', 'all', 'random', 'fixed'],help='Mask mode: default=use original masks in dataset (can be multiple irregular regions); '
    #                          'all=entire image as mask region; '
    #                          'random=use original mask total area as base, take square root to get square side length, top-left position random; '
    #                          'fixed=same as above but top-left position fixed at (0,0)')

    # IAPR-TC parameters
    # parser = argparse.ArgumentParser(description="Image poisoning attack based on critical regions")
    # parser.add_argument("--target_path", default='./data/IAPR-TC/images/31/31474.jpg', help='Target image path, all source images are poisoned to be close to this target')
    # parser.add_argument("--poison_data_last_name", default='110_207_31474_with_cosine_eps=4_neg_weight=0.02', help='Poisoned image suffix, used to distinguish datasets with different operations, e.g. switching loss function')
    # parser.add_argument("--neg_contrast_weight", type=float, default=0.01, help='Negative contrastive loss weight: based on BadCLIP paper, keep poisoned images away from original images to improve ASR, 0 means disabled')
    # parser.add_argument("--eps", type=float, default=8/255, help='L∞ constraint strength: maximum pixel modification, default 4/255≈1.6%, smaller is more stealthy')
    # parser.add_argument('--split', default='test', type=str, choices=['train', 'test', 'database'], help='Dataset split: train/test/database')
    # parser.add_argument('--ratio', type=float, default=1, help='Data subset ratio: e.g. 0.01 means use 1% of data for poisoning')
    # parser.add_argument('--dataset', type=str, default='IAPR-TC', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='Dataset name: supported benchmark datasets')
    # parser.add_argument("--iter_attack", type=int, default=2000, help='PGD iteration count: controls attack strength, more iterations = stronger attack but slower')
    # parser.add_argument("--lr_attack", type=float, default=1.5, help='PGD learning rate: controls step size per iteration')
    # # parser.add_argument("--tv_weight", type=float, default=0.00004, help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence, for L2')
    # parser.add_argument("--tv_weight", type=float, default=0.000004, help='Total variation regularization weight: controls smoothness, larger is smoother but slower convergence')
    # parser.add_argument("--poison_save_pth", default=None, help='Poisoned image save path. If None, use data/{dataset}/poisoned_image/{split}_{ratio}}')
    # parser.add_argument('--device', default='0', type=str, help='GPU device ID: e.g. "0" uses first GPU, "cpu" uses CPU')
    # parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32', type=str, help='CLIP pretrained model path: used for embedding space computation')
    # parser.add_argument('--data_path', default='./data', type=str, help='Dataset root directory path')
    # parser.add_argument("--batch_size", type=int, default=128, help='Batch size: larger is faster but uses more GPU memory')
    # parser.add_argument("--use_amp", type=bool, default=True, help='Enable AMP mixed precision training, can significantly improve training speed (requires GPU supporting FP16)')
    # parser.add_argument("--mask_mode", type=str, default='default',choices=['default', 'all', 'random', 'fixed'],help='Mask mode: default=use original masks in dataset (can be multiple irregular regions); '
    #                          'all=entire image as mask region; '
    #                          'random=use original mask total area as base, take square root to get square side length, top-left position random; '
    #                          'fixed=same as above but top-left position fixed at (0,0)')

    args = parser.parse_args()
    # ===== Print all parameters =====
    logger.info("=" * 60)
    logger.info("All parameter configurations:")
    logger.info("=" * 60)
    for arg in vars(args):
        value = getattr(args, arg)
        # If float, keep appropriate precision
        if isinstance(value, float):
            if value < 1:
                logger.info(f"  {arg}: {value:.6f}")
            else:
                logger.info(f"  {arg}: {value}")
        else:
            logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)

    return args

def load_image(image_path):
    """Load image"""
    return Image.open(image_path).convert('RGB')

def load_image_tensors_with_masks(target_path, dataset_info, img_size):
    """
    Load image tensors and corresponding critical region masks
    This function is the core of data loading, responsible for:
    1. Read image path list from cm_{split}_imgs.txt
    2. Load corresponding original images, target image and binary masks
    3. Resize all images to uniform size to ensure tensor operation consistency
    4. Return data usable for batch PGD attack

    Args:
        target_path (str): target image path, containing a single target.jpg file
        dataset_info (dict): dataset information dict, must contain:
            - 'dataset': dataset name
            - 'split': data split (train/test/database)
            - 'ratio': data subset ratio (0-1)
            - 'data_path': dataset root directory path
        img_size (int): uniform image scaling size (CLIP model standard size is 224)

    Returns:
        tuple: (images_base, images_target, masks, valid_image_ids, valid_image_rel_paths)
            - images_base: base image tensors [N, 3, H, W], N is number of images
            - images_target: target image tensors [N, 3, H, W], all images use the same target
            - masks: binary mask tensors [N, 3, H, W], 0=non-critical region, 1=critical region
            - valid_image_ids: list of valid image IDs, used for subsequent save naming
            - valid_image_rel_paths: list of valid image relative paths, used to preserve directory structure

    Raises:
        ValueError: raised when required files cannot be found
    """
    images_base = []
    images_target = []
    masks = []
    valid_image_ids = []
    valid_image_rel_paths = []
    # ===== Step 1: Read image path list =====
    logger.info("Reading image path list...")
    # Read image relative paths from standard dataset file
    # File format: one relative path per line, e.g. "images/20/20045.jpg"
    img_list_file = os.path.join(dataset_info['data_path'], dataset_info['dataset'],
                                f'cm_{dataset_info["split"]}_imgs.txt')

    logger.info(f"Image list file path: {img_list_file}")
    if not os.path.exists(img_list_file):
        raise ValueError(f"Image list file does not exist: {img_list_file}")

    # Read all image relative paths
    logger.info("Reading image list file...")
    with open(img_list_file, 'r') as f:
        image_rel_paths = [line.strip() for line in f if line.strip()]

    logger.info(f"Total {len(image_rel_paths)} image relative paths read")

    # Only process first ratio portion of data (for experiment control of data volume)
    subset_size = int(len(image_rel_paths) * dataset_info['ratio'])
    image_rel_paths = image_rel_paths[:subset_size]

    logger.info(f"Using data subset ratio {dataset_info['ratio']:.4f}, processing first {subset_size} images")

    # ===== Step 2: GPU setup and preload target image =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    logger.info("Loading target image...")
    target_image_path = target_path
    logger.info(f"Target image path: {target_image_path}")

    if not os.path.exists(target_image_path):
        raise ValueError(f"Target image does not exist: {target_image_path}")

    target_img = load_image(target_image_path)
    target_tensor = _TO_TENSOR(_RESIZE_224(target_img)).to(device)
    logger.info(f"Target image loaded, device: {target_tensor.device}")

    import time
    _t0 = time.time()
    logger.info(f"Starting to process {len(image_rel_paths)} images...")

    stat_check_file = []
    stat_load_base = []
    stat_load_mask = []
    stat_tensor_ops = []

    for idx, image_rel_path in enumerate(image_rel_paths):
        if idx % 100 == 0:  # print progress every 100 images
            elapsed = time.time() - _t0
            avg_ms = elapsed / max(idx, 1) * 1000
            logger.info(f"Processing image {idx+1}/{len(image_rel_paths)}... (average {avg_ms:.1f}ms per image)")

        # Parse relative path, extract image ID
        # e.g.: "images/20/20045.jpg" -> image ID is "20045"
        image_id = os.path.splitext(os.path.basename(image_rel_path))[0]

        # Build file paths
        # Original image path: data/{dataset}/images/... (keep original relative path)
        image_path = os.path.join(dataset_info['data_path'], dataset_info['dataset'], image_rel_path)

        # Mask path: replace "images" with "masks_image/{split}_{ratio}"
        # ratio_str = f"{dataset_info['ratio']:.4f}"
        # subdir = f"{dataset_info['split']}_{ratio_str}"
        subdir = f"{dataset_info['split']}"
        # MS-COCO compatibility: replace first segment train2014/val2014 with images
        if dataset_info['dataset'] == 'MS-COCO':
            # Only replace first folder name in path
            parts = image_rel_path.split('/', 1)
            if len(parts) == 2:
                image_rel_path = f'images/{parts[1]}'
        mask_rel_path = image_rel_path.replace('images', f'masks_image/{subdir}', 1)
        mask_path = os.path.join(dataset_info['data_path'], dataset_info['dataset'], mask_rel_path)
        mask_path = mask_path.replace('jpg', 'png')

        # Check if required files exist
        t_check = time.time()
        if not os.path.exists(image_path):
            logger.info(f"Warning: image file not found {image_path}, skipping")
            continue
        if not os.path.exists(mask_path):
            logger.info(f"Warning: mask file not found {mask_path}, skipping")
            continue
        stat_check_file.append(time.time() - t_check)

        t_loop = time.time()
        try:
            # Load original image as base image
            t_load = time.time()
            base_img = load_image(image_path)
            stat_load_base.append(time.time() - t_load)

            # Load binary mask image
            t_mask = time.time()
            mask_img = load_image(mask_path)
            mask_array = np.array(mask_img)
            stat_load_mask.append(time.time() - t_mask)

            # Process binary mask image (0-255 range)
            if len(mask_array.shape) == 3:
                # If RGB, convert to grayscale, all three channels have same value, just take first channel, no need to average across channels
                mask_gray = mask_array[..., 0]  # take first channel
                # mask_gray = np.mean(mask_array, axis=2)
            else:
                mask_gray = mask_array

            # Normalize to 0-1 range (already binary, 255 means critical region)
            mask_binary = (mask_gray / 255.0).astype(np.float32)

            # Expand to 3 channels
            mask_binary = np.stack([mask_binary] * 3, axis=2)

            # Resize (all images and masks resized to same size)
            t_ops = time.time()
            # Directly use numpy normalization + resize, avoid duplicate creation of PIL ToTensor
            base_np = np.array(_RESIZE_224(base_img)).astype(np.float32) / 255.0
            base_tensor = torch.from_numpy(base_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]

            # Mask processing: first resize with numpy, then process with torch
            mask_resized = np.array(mask_img.resize((img_size, img_size), Image.NEAREST))
            if len(mask_resized.shape) == 3:
                mask_resized = mask_resized[..., 0]
            mask_tensor = torch.from_numpy(mask_resized / 255.0).float().unsqueeze(0).to(device)  # [1, H, W]
            mask_tensor = mask_tensor.expand(1, 3, -1, -1)  # [1, 3, H, W]

            images_base.append(base_tensor)
            images_target.append(target_tensor.clone().unsqueeze(0))
            masks.append(mask_tensor)
            valid_image_ids.append(image_id)
            valid_image_rel_paths.append(image_rel_path)
            stat_tensor_ops.append(time.time() - t_ops)

        except Exception as e:
            logger.info(f"Error loading image {image_id}: {e}")
            continue

        t_this = time.time() - t_loop
        # logger.info(f"[{idx+1}/{len(image_rel_paths)}] {image_id} | check: {stat_check_file[-1]*1000:.1f}ms, load_base: {stat_load_base[-1]*1000:.1f}ms, load_mask: {stat_load_mask[-1]*1000:.1f}ms, tensor_ops: {stat_tensor_ops[-1]*1000:.1f}ms, this_iter: {t_this*1000:.1f}ms")

    if len(images_base) == 0:
        raise ValueError("No valid image-mask pairs found")

    images_base = torch.cat(images_base, axis=0)
    images_target = torch.cat(images_target, axis=0)
    masks = torch.cat(masks, axis=0)

    logger.info(f"Successfully loaded {images_base.shape[0]} image-mask pairs")
    total_time = time.time() - _t0
    n = len(stat_check_file)
    if n > 0:
        logger.info(f"[Timing summary] Total time: {total_time:.1f}s | "
                    f"File check: mean={np.mean(stat_check_file)*1000:.1f}ms, max={np.max(stat_check_file)*1000:.1f}ms | "
                    f"Load base: mean={np.mean(stat_load_base)*1000:.1f}ms, max={np.max(stat_load_base)*1000:.1f}ms | "
                    f"Load mask: mean={np.mean(stat_load_mask)*1000:.1f}ms, max={np.max(stat_load_mask)*1000:.1f}ms | "
                    f"Tensor ops: mean={np.mean(stat_tensor_ops)*1000:.1f}ms, max={np.max(stat_tensor_ops)*1000:.1f}ms")
    return images_base, images_target, masks, valid_image_ids, valid_image_rel_paths


class PairedImageDatasetWithMasks(torch.utils.data.Dataset):
    """
    Dataset class containing base images, target images and corresponding critical region masks
    """
    def __init__(self, images_base, images_target, masks):
        super().__init__()
        assert images_base.shape[0] == images_target.shape[0] == masks.shape[0]
        self.images_base = images_base
        self.images_target = images_target
        self.masks = masks

    def __len__(self):
        return self.images_base.shape[0]

    def __getitem__(self, index):
        return self.images_base[index], self.images_target[index], self.masks[index]


# CLIP model normalization (same as poison_llava.py)
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def get_clip_model(from_pretrain):
    """
    Load CLIP model and processor (same as image_region_extractor.py)
    """
    model = CLIPModel.from_pretrained(from_pretrain)
    processor = CLIPProcessor.from_pretrained(from_pretrain)
    model.eval()
    return model, processor


def total_variation_loss(x, mask):
    """
    Compute total variation loss, used to reduce artifacts and high-frequency noise
    Only compute TV loss within mask region
    """
    # Only compute TV loss on mask region
    x_masked = x * mask

    # Compute horizontal and vertical gradients
    tv_h = torch.abs(x_masked[:, :, :, :-1] - x_masked[:, :, :, 1:])
    tv_v = torch.abs(x_masked[:, :, :-1, :] - x_masked[:, :, 1:, :])

    # Only compute TV in mask corresponding region
    mask_h = mask[:, :, :, :-1]
    mask_v = mask[:, :, :-1, :]

    # Originally a scalar loss summed over all samples, now a batch vector
    # tv_loss = (tv_h * mask_h).sum() + (tv_v * mask_v).sum()
    # Sum over channel and spatial dimensions for each sample, keeping batch dimension, each sample's TV loss independent.
    tv_loss_per_sample = (tv_h * mask_h).sum(dim=[1,2,3]) + (tv_v * mask_v).sum(dim=[1,2,3])
    return tv_loss_per_sample


def get_poisoned_mask(masks, mode='default'):
    """
    Transform masks according to mask_mode.

    Consistent with same-named function in visual.py, and supports:
    - 'default': directly return original mask (may contain multiple irregular regions)
    - 'all'    : entire image region as mask (all 1)
    - 'random' : use original mask total area as base, take square root to get square side length l,
                 position randomly sampled within image range
    - 'fixed'  : same total area base, take square root to get square side length l,
                 position fixed at (0, 0)

    Args:
        masks: tensor of shape [N, C, H, W], C usually 3
        mode : 'default' / 'all' / 'random' / 'fixed'

    Returns:
        New mask tensor with same shape, dtype, device as masks
    """
    if mode == 'default':
        return masks

    if mode == 'all':
        return torch.ones_like(masks)

    if mode not in ('random', 'fixed'):
        raise ValueError("Unknown mask mode {}".format(mode))

    _, _, height, width = masks.size()
    new_masks = torch.zeros(size=masks.size(), dtype=masks.dtype, device=masks.device)
    logger.info("[mask] mode={}, image={}x{}, N={}".format(mode, height, width, masks.size(0)))
    for i, mask in enumerate(masks):
        mask_single = mask[0]
        size = int((mask_single > 0.5).sum().item())
        if size <= 0:
            if i < 10:
                logger.info("[mask]   sample {:4d}: area=0 -> skip".format(i))
            continue

        l = min(int(size ** 0.5), height, width)
        if mode == 'random':
            x = np.random.randint(0, width - l + 1)
            y = np.random.randint(0, height - l + 1)
        else:  # mode == 'fixed'
            x, y = 0, 0

        new_masks[i, :, y:y + l, x:x + l] = 1
        if i < 10:
            logger.info("[mask]   sample {:4d}: area={} -> l={}, (x,y)=({},{})".format(i, size, l, x, y))
    logger.info("[mask] done, avg coverage={:.4f}".format(new_masks[:, 0].mean().item()))
    return new_masks


def embedding_attack_Linf_with_mask(clip_model, image_base, image_victim, mask, emb_dist,
                                   iters=100, lr=1/255, eps=8/255, tv_weight=0.001, diff_aug=None, resume_X_adv=None,
                                   use_amp=False, neg_contrast_weight=0.0):
    """
    Constrained PGD attack algorithm based on critical region masks

    This function implements the core algorithm for restricted adversarial attack within critical image regions:
    - Goal: minimize ||emb(poison) - emb(target)||₂
    - Constraint: only modify pixels within mask-marked critical regions
    - Method: iterative optimization using projected gradient descent (PGD)

    Args:
        clip_model: CLIP vision model, used to compute image embeddings
        image_base: base image tensors [batch_size, 3, H, W], original images to be poisoned
        image_victim: target image tensors [batch_size, 3, H, W], poisoning target
        mask: binary mask tensors [batch_size, 3, H, W], 1=critical region (modifiable), 0=non-critical region (unchanged)
        emb_dist: embedding distance computation function, usually L2_norm
        iters: total PGD iterations, controls attack strength
        lr: learning rate, step size for each gradient update
        eps: L∞ constraint range, maximum pixel modification (default 8/255≈3%)
        diff_aug: differentiable data augmentation function (optional, for improving attack robustness)
        resume_X_adv: resumed adversarial example tensors (optional, for continuing previous attack)
        use_amp: whether to enable AMP mixed precision training (optional, default False)

    Returns:
        tuple: (X_adv_best, final_loss)
            - X_adv_best: best adversarial examples [batch_size, 3, H, W]
            - final_loss: final embedding distance loss [batch_size]

    Algorithm flow:
    1. Initialize adversarial examples (add random noise to base images)
    2. Compute target embeddings (fixed)
    3. PGD iterative optimization:
       - Compute current sample embeddings
       - Compute distance loss to target embeddings
       - Apply gradient update only in mask region
       - Project back to L∞ constraint space
    4. Return best result
    """
    device = image_base.device
    bs = image_base.size(0)

    # ===== Option B: remove GradScaler, pure FP16 computation (vision_model already converted to FP16) =====
    scaler = None  # No longer need GradScaler: input is already FP16, no dynamic scaling needed

    # ===== Option A companion: auto-detect vision_model dtype to decide input dtype =====
    use_fp16 = (use_amp and torch.cuda.is_available())
    vision_is_fp16 = False
    if use_fp16:
        try:
            # Probe dtype using vision_model's first parameter — compatible with transformers/open_clip/clip
            probe = next(clip_model.vision_model.parameters())
            vision_is_fp16 = (probe.dtype == torch.float16)
        except Exception:
            vision_is_fp16 = False

    def _maybe_to_fp16(x):
        """If vision_model is FP16, convert input to FP16 (avoid dtype mismatch)"""
        if vision_is_fp16 and x.dtype != torch.float16:
            return x.half()
        return x

    with torch.no_grad():
        embedding_targets = clip_model.vision_model(_maybe_to_fp16(normalize(image_victim))).pooler_output
        # If vision_model is FP16, output is also FP16, convert back to FP32 to avoid subsequent loss computation overflow
        if vision_is_fp16:
            embedding_targets = embedding_targets.float()
        # Precompute embeddings of original clean images (for negative contrastive loss)
        embedding_bases = clip_model.vision_model(_maybe_to_fp16(normalize(image_base))).pooler_output
        if vision_is_fp16:
            embedding_bases = embedding_bases.float()

    # Initialize adversarial examples
    noise = (torch.rand(*image_base.shape) * 2 * eps - eps).to(device)
    X_adv = image_base.clone().detach() + noise * mask
    if resume_X_adv is not None:
        logger.info('Resuming from a given X_adv')
        X_adv = resume_X_adv.clone().detach()
    X_adv.data = X_adv.data.clamp(0, 1)
    X_adv.requires_grad_(True)

    # Use AdamW optimizer instead of SGD, more stable for complex objective functions
    # AdamW with weight decay, better generalization
    optimizer = optim.AdamW([X_adv], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Use cosine annealing scheduler, smoother than MultiStepLR
    # T_max=iters makes learning rate gradually decrease over all iterations
    # eta_min=lr*0.01 final learning rate not lower than 1% of initial
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=lr * 0.01)

    loss_best = 1e8 * torch.ones(bs).to(device)
    X_adv_best = X_adv.clone().detach()
    loss_history = []  # record loss at each iteration

    # Initialize early stopping parameters
    best_avg_loss = float('inf')
    patience = 50 # if no improvement for 100 rounds, stop
    # For L2 loss it is 0.002
    # min_improve = 0.01      # minimum improvement threshold
    min_improve = 0.0005      # minimum improvement threshold
    no_improve_count = 0

    for i in tqdm(range(iters), desc="PGD poisoning attack", mininterval=3):
        if diff_aug is not None:
            X_adv_input_to_model = _maybe_to_fp16(normalize(diff_aug(X_adv)))
        else:
            X_adv_input_to_model = _maybe_to_fp16(normalize(X_adv))

        # ===== Option A companion: vision_model already converted to FP16, direct forward, no autocast needed =====
        # Since input is already FP16, autocast is not only meaningless but also introduces extra dtype check overhead
        embedding_adv = clip_model.vision_model(X_adv_input_to_model).pooler_output
        # Convert back to FP32 to avoid emb_dist overflow in FP16
        if vision_is_fp16:
            embedding_adv_for_loss = embedding_adv.float()
        else:
            embedding_adv_for_loss = embedding_adv

        embedding_loss = emb_dist(embedding_adv_for_loss, embedding_targets)
        # Add total variation regularization to reduce artifacts
        tv_loss = total_variation_loss(X_adv, mask)
        # Add negative contrastive loss (based on BadCLIP paper, computed using emb_dist)
        # neg_loss = 1 - cosine_sim(poison, clean), need to maximize it i.e. keep poisoned image away from original image
        if neg_contrast_weight > 0:
            neg_loss = emb_dist(embedding_adv_for_loss, embedding_bases)
            loss = embedding_loss + tv_weight * tv_loss - neg_contrast_weight * neg_loss
        else:
            loss = embedding_loss + tv_weight * tv_loss

        # Record history
        current_avg_loss = loss.mean().item()
        loss_history.append(current_avg_loss)  # record loss

        # Update best result
        if i % max(int(iters/1000), 1) == 0:
            with torch.no_grad():
                mask_loss = (loss < loss_best)
                if mask_loss.sum() > 0:
                    indices = torch.where(mask_loss)[0]
                    loss_best[indices] = loss[indices].clone().detach()
                    X_adv_best[indices] = X_adv[indices].clone().detach()

        # Compute total loss for backpropagation
        # Use mean() instead of sum(): makes loss independent of batch_size
        # This way learning rate can be set independently of batch_size, more stable
        total_loss = loss.mean()
        optimizer.zero_grad()

        # ===== Option B: direct backpropagation, remove GradScaler's 5 GPU↔CPU synchronizations =====
        total_loss.backward()
        # Apply gradient update only in mask region (set gradient outside mask to zero)
        X_adv.grad = X_adv.grad * mask
        optimizer.step()

        scheduler.step()

        if i % max(int(iters/20), 1) == 0:
            logger.info('\n')
            logger.info(f'embedding_loss mean: {embedding_loss.mean().item():.4f}')
            logger.info(f'tv_loss mean: {tv_loss.mean().item():.4f}')
            logger.info(f'tv_weight * tv_loss mean: {(tv_weight * tv_loss).mean().item():.4f}')
            if neg_contrast_weight > 0:
                logger.info(f'neg_loss mean: {neg_loss.mean().item():.4f}')
                logger.info(f'neg_contrast_weight * neg_loss mean: {(neg_contrast_weight * neg_loss).mean().item():.4f}')
            logger.info(f'Iter {i}: loss={current_avg_loss:.4f}, lr={scheduler.get_last_lr()[0]*255:.4f}')

        # Project to constraint space: restrict perturbation in mask region, keep original image in non-mask region
        mask_bool = mask > 0.5
        with torch.no_grad():
            X_adv.data = torch.where(mask_bool,
                                    torch.clamp(X_adv, image_base - eps, image_base + eps),
                                    image_base)
            X_adv.data = X_adv.data.clamp(0, 1)
            X_adv.grad = None

        # Debug: verify projection works correctly (print at iteration 0 and last iteration)
        if i % 200 == 0:
            with torch.no_grad():
                actual_perturbation = (X_adv - image_base).abs()
                max_perturb = actual_perturbation.max().item()
                # Count perturbations inside and outside mask separately
                mask_region_perturb = (actual_perturbation * mask).max().item()
                non_mask_region_perturb = (actual_perturbation * (1 - mask)).max().item()
                logger.info(f'Iter {i}: max_perturb={max_perturb:.4f} (mask: {mask_region_perturb:.4f}, non_mask: {non_mask_region_perturb:.4f}), eps={eps:.4f}')
        # Early stopping logic
        if i > 100:
            # Determine if there is significant improvement
            if current_avg_loss < best_avg_loss - min_improve:
                best_avg_loss = current_avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If no improvement for patience consecutive rounds, trigger early stopping
            if no_improve_count >= patience:
                logger.info('\n')
                logger.info(f"[Early stopping triggered] Iter {i}: loss reduction below {min_improve} for {patience} consecutive rounds, stopping optimization early.")
                break

        if torch.isnan(total_loss):
            logger.info(f'Encounter nan loss at iteration {i}')
            break

    with torch.no_grad():
        if diff_aug:
            X_adv_best_input = _maybe_to_fp16(normalize(diff_aug(X_adv_best)))
        else:
            X_adv_best_input = _maybe_to_fp16(normalize(X_adv_best))
        embedding_final = clip_model.vision_model(X_adv_best_input).pooler_output
        if vision_is_fp16:
            embedding_final = embedding_final.float()
        final_loss = emb_dist(embedding_final, embedding_targets)
        logger.info(f'Best Total loss: {final_loss.mean().item():.4f}')

    return X_adv_best, final_loss.detach(), loss_history


def get_clip_embedding_from_pil(img, clip_model, device, resize_transform, to_tensor_transform, normalize_transform):
    """
    General: get CLIP embedding from PIL image (completely consistent with training preprocessing)

    Args:
        img: PIL image
        clip_model: CLIP model
        device: device where model is located
        resize_transform: predefined Resize transform
        to_tensor_transform: predefined ToTensor transform
        normalize_transform: predefined Normalize transform

    Returns:
        embedding: CLIP embedding vector of single image
    """
    # 1. Uniform Resize method (completely consistent with training)
    img_resized = resize_transform(img)
    # 2. Convert to Tensor and add batch dimension
    img_tensor = to_tensor_transform(img_resized).unsqueeze(0).to(device)
    # 3. Normalize
    img_norm = normalize_transform(img_tensor)
    # 4. Forward propagation (automatically adapt to FP16 input)
    with torch.no_grad():
        try:
            probe = next(clip_model.vision_model.parameters())
            vision_is_fp16 = (probe.dtype == torch.float16)
        except Exception:
            vision_is_fp16 = False
        img_input = img_norm.half() if vision_is_fp16 else img_norm
        embedding = clip_model.vision_model(img_input).pooler_output
        if vision_is_fp16:
            embedding = embedding.float()
    return embedding

def L2_norm(a, b):
    """
    Compute L2 distance
    """
    assert a.size(0) == b.size(0), 'two inputs contain different number of examples'
    bs = a.size(0)
    dist_vec = (a - b).view(bs, -1).norm(p=2, dim=1)
    return dist_vec

def cosine_similarity_loss(a, b):
    """1 - cosine similarity, smaller is better"""
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return 1 - (a_norm * b_norm).sum(dim=1)


def save_poison_data(images_to_save, image_ids, image_rel_paths, save_path, dataset_info, poison_data_last_name,
                    num_workers=None):
    """
    Save poisoned image data to disk (multi-threaded concurrent version)

    Args:
        images_to_save: poisoned image tensors [N, 3, H, W]
        image_ids: list of image IDs
        image_rel_paths: list of image relative paths
        save_path: save directory path
        dataset_info: dataset information dict
        poison_data_last_name: poisoned dataset suffix
        num_workers: number of disk write threads (auto if not specified: min(cpu*2, 16, N))
    """
    assert len(images_to_save.size()) == 4, 'images_to_save should be a batch of image tensors'
    assert len(images_to_save) == len(image_ids), 'images_to_save and image_ids must have the same length'
    assert len(images_to_save) == len(image_rel_paths), 'images_to_save and image_rel_paths must have the same length'

    num_total = len(images_to_save)
    if num_workers is None:
        # Disk write is IO-intensive (CPU idle while thread waits for disk), use 2*cpu_count(); cap at 16 to prevent cloud server thread explosion
        num_workers = min((os.cpu_count() or 4) * 2, 16, max(num_total, 1))

    # Get replacement directory
    ratio_str = f"{dataset_info['ratio']:.4f}"
    subdir = f"{dataset_info['split']}_{ratio_str}_{poison_data_last_name}"
    replaced_dir = f'poisoned_image/{subdir}'

    # Precompute all save paths (CPU-side serial)
    save_paths = []
    for i in range(num_total):
        split_idx = image_rel_paths[i].find('/')
        poison_filename = replaced_dir + image_rel_paths[i][split_idx:]
        img_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], poison_filename)
        os.makedirs(os.path.dirname(img_pth), exist_ok=True)
        save_paths.append(img_pth)

    # Move tensors to CPU in advance (avoid cross-device copy inside worker)
    images_cpu = images_to_save.detach().cpu()
    # Split into list to hand to thread pool
    tasks = list(zip(images_cpu, save_paths, image_ids))

    def _write_one(img, path, image_id):
        save_image(img, path)
        return image_id

    # Multi-threaded concurrent disk write
    n_workers = min(int(num_workers), num_total) if num_total > 0 else 1
    logger.info(f"Starting concurrent saving of {num_total} poisoned images (workers={n_workers})")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(lambda t: _write_one(*t), tasks),
                  total=len(tasks), desc="Saving poisoned images", mininterval=1))

    logger.info(f'Saved {num_total} poisoned images to {save_path}')


def calculate_image_quality_metrics(original_images, poisoned_images):
    """
    Compute image quality metrics: PSNR and SSIM
    Used to evaluate visual stealthiness of poisoning attack
    """
    try:
        import torch.nn.functional as F
        from skimage.metrics import structural_similarity as ssim

        psnr_values = []
        ssim_values = []

        for i in range(len(original_images)):
            orig = original_images[i].cpu().numpy().transpose(1, 2, 0)
            poison = poisoned_images[i].cpu().numpy().transpose(1, 2, 0)

            # Compute PSNR
            mse = np.mean((orig - poison) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

            # Compute SSIM (per channel then average)
            ssim_val = 0
            for c in range(3):
                ssim_val += ssim(orig[:, :, c], poison[:, :, c], data_range=1.0)
            ssim_val /= 3
            ssim_values.append(ssim_val)

        psnr_values = np.array(psnr_values)
        ssim_values = np.array(ssim_values)

        # Compute MSE values for printing
        mse_values = np.array([np.mean((original_images[i].cpu().numpy() - poisoned_images[i].cpu().numpy()) ** 2)
                              for i in range(len(original_images))])

        logger.info("\n=== Visual quality evaluation ===")
        logger.info(f"Average MSE: {mse_values.mean():.6f} (range: {mse_values.min():.6f} - {mse_values.max():.6f})")
        logger.info(f"Average PSNR: {psnr_values.mean():.2f} dB (range: {psnr_values.min():.2f} - {psnr_values.max():.2f})")
        logger.info(f"Average SSIM: {ssim_values.mean():.4f} (range: {ssim_values.min():.4f} - {ssim_values.max():.4f})")
        logger.info(f"MSE std: {mse_values.std():.6f}")
        logger.info(f"PSNR std: {psnr_values.std():.2f}")
        logger.info(f"SSIM std: {ssim_values.std():.4f}")
        return psnr_values.mean(), ssim_values.mean()

    except ImportError:
        logger.info("Warning: cannot compute SSIM metric, please install scikit-image: pip install scikit-image")
        # Only compute PSNR
        psnr_values = []
        for i in range(len(original_images)):
            orig = original_images[i].cpu().numpy()
            poison = poisoned_images[i].cpu().numpy()

            mse = np.mean((orig - poison) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            psnr_values.append(psnr)

        psnr_values = np.array(psnr_values)
        # Compute MSE values for printing
        mse_values = np.array([np.mean((original_images[i].cpu().numpy() - poisoned_images[i].cpu().numpy()) ** 2)
                              for i in range(len(original_images))])

        logger.info("\n=== Visual quality evaluation ===")
        logger.info(f"Average MSE: {mse_values.mean():.6f} (range: {mse_values.min():.6f} - {mse_values.max():.6f})")
        logger.info(f"Average PSNR: {psnr_values.mean():.2f} dB (range: {psnr_values.min():.2f} - {psnr_values.max():.2f})")
        logger.info(f"MSE std: {mse_values.std():.6f}")
        logger.info(f"PSNR std: {psnr_values.std():.2f}")
        return psnr_values.mean(), 0.0


def calculate_psnr_ssim(original_images, poisoned_images):
    """Compute PSNR and SSIM (using torchmetrics library, GPU accelerated)"""
    # Compute MSE (per sample)
    mse_values = ((original_images - poisoned_images) ** 2).mean(dim=(1, 2, 3))
    mse_mean = mse_values.mean()

    # Compute PSNR (using torchmetrics, supports GPU)
    psnr = peak_signal_noise_ratio(original_images, poisoned_images)

    # Compute SSIM (using torchmetrics, supports GPU)
    ssim = structural_similarity_index_measure(original_images, poisoned_images)

    # Log output
    logger.info("\n=== Visual quality evaluation ===")
    logger.info(f"Average MSE: {mse_mean:.6f} (range: {mse_values.min():.6f} - {mse_values.max():.6f})")
    logger.info(f"Average PSNR: {psnr:.2f} dB")
    logger.info(f"Average SSIM: {ssim:.4f}")

    return psnr, ssim



def test_attack_efficacy(clip_model, target_path, dataset_info, valid_image_ids, save_path, device='cuda', sample_num=50, emb_dist=L2_norm, images_base_tensor=None, X_adv_tensor=None):
    """
    Test attack effectiveness, including embedding space distance and visual quality metrics (optimized version: batch processing)

    New parameters:
        images_base_tensor: original image tensors [N, 3, H, W], if None load from disk
        X_adv_tensor: poisoned image tensors [N, 3, H, W], if None load from disk
    """
    tensors_base, tensors_poison = None, None

    # If in-memory tensors provided, use directly
    if images_base_tensor is not None and X_adv_tensor is not None:
        logger.info("Using in-memory tensors to compute quality metrics...")
        tensors_base = images_base_tensor[:sample_num].to(device)
        tensors_poison = X_adv_tensor[:sample_num].to(device)

        # Use in-memory tensors to compute PSNR/SSIM (GPU accelerated)
        avg_psnr, avg_ssim = calculate_psnr_ssim(tensors_base.cpu(), tensors_poison.cpu())

        # Compute pixel distance
        pixel_dist = (tensors_base - tensors_poison).abs().flatten(1).max(dim=1)[0]
        logger.info('\n=== Attack effectiveness verification ===')
        logger.info(f'Max pixel distance (x255): {(pixel_dist.max().item() * 255):.4f}')
        logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
        logger.info(f"Average SSIM: {avg_ssim:.4f}")

        return avg_psnr, avg_ssim

    # Original logic: load images from disk
    logger.info("Loading images from disk for verification...")

    # Re-read path info from cm_{split}_imgs.txt to build base image paths
    img_list_file = os.path.join(dataset_info['data_path'], dataset_info['dataset'],
                                f'cm_{dataset_info["split"]}_imgs.txt')

    with open(img_list_file, 'r') as f:
        all_image_rel_paths = [line.strip() for line in f if line.strip()]

    # Build image path lists
    image_paths_base = []
    image_paths_poison = []

    for i in range(min(len(valid_image_ids), sample_num)):
        image_id = valid_image_ids[i]

        # Find corresponding relative path from image list
        image_rel_path = None
        for rel_path in all_image_rel_paths:
            if os.path.splitext(os.path.basename(rel_path))[0] == image_id:
                image_rel_path = rel_path
                break

        if image_rel_path is None:
            continue

        image_base_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], image_rel_path)
        split_idx = image_rel_path.find('/')
        image_poison_pth = save_path + image_rel_path[split_idx:]

        image_paths_base.append(image_base_pth)
        image_paths_poison.append(image_poison_pth)

    n_samples = len(image_paths_base)
    if n_samples == 0:
        logger.info("No valid images found for verification")
        return 0.0, 0.0

    logger.info(f"Batch loading {n_samples} images for verification...")

    # ===== Batch load images =====
    images_base = [load_image(p) for p in image_paths_base]
    images_poison = [load_image(p) for p in image_paths_poison]
    target_img = load_image(target_path)

    # ===== Batch preprocessing: complete resize and tensor conversion for all images at once =====
    logger.info("Batch preprocessing images...")
    tensors_base = torch.stack([_TO_TENSOR(_RESIZE_224(img)) for img in images_base]).to(device)
    tensors_poison = torch.stack([_TO_TENSOR(_RESIZE_224(img)) for img in images_poison]).to(device)
    tensor_target = _TO_TENSOR(_RESIZE_224(target_img)).unsqueeze(0).to(device)

    # ===== Batch compute embeddings (key optimization) =====
    logger.info("Batch computing CLIP embeddings...")

    # Probe vision_model dtype
    try:
        probe = next(clip_model.vision_model.parameters())
        vision_is_fp16 = (probe.dtype == torch.float16)
    except Exception:
        vision_is_fp16 = False

    # Batch normalization
    norm_base = normalize(tensors_base)
    norm_poison = normalize(tensors_poison)
    norm_target = normalize(tensor_target)
    if vision_is_fp16:
        norm_base = norm_base.half()
        norm_poison = norm_poison.half()
        norm_target = norm_target.half()

    # Compute all embeddings at once
    with torch.no_grad():
        emb_base = clip_model.vision_model(norm_base).pooler_output
        emb_poison = clip_model.vision_model(norm_poison).pooler_output
        emb_target = clip_model.vision_model(norm_target).pooler_output
        if vision_is_fp16:
            emb_base = emb_base.float()
            emb_poison = emb_poison.float()
            emb_target = emb_target.float()

    # ===== Compute distances =====
    dist_base_target = emb_dist(emb_base, emb_target)
    dist_poison_target = emb_dist(emb_poison, emb_target)

    # Compute pixel distance (using GPU)
    pixel_dist = (tensors_base - tensors_poison).abs().flatten(1).max(dim=1)[0]

    logger.info('\n=== Attack effectiveness verification ===')
    logger.info(f'Embedding distance between original images and target image: {dist_base_target.mean().item():.4f}')
    logger.info(f'Embedding distance between poisoned images and target image: {dist_poison_target.mean().item():.4f}')
    logger.info(f'Distance reduction ratio: {(dist_base_target - dist_poison_target).mean().item() / dist_base_target.mean().item():.4f}')
    logger.info(f'Max pixel distance (x255): {(pixel_dist.max().item() * 255):.4f}')

    # ===== Compute quality metrics =====
    avg_psnr, avg_ssim = calculate_psnr_ssim(tensors_base.cpu(), tensors_poison.cpu())

    return avg_psnr, avg_ssim


def main():
    """
    Main function: execute complete image poisoning attack workflow

    Execution steps:
    1. Argument parsing and environment setup
    2. Model loading (CLIP vision encoder)
    3. Data loading (images, masks, target image)
    4. PGD attack execution (batch processing)
    5. Result saving
    6. Attack effectiveness verification
    """
    logger.info("=== Starting image poisoning attack ===")
    # ===== [New] Fix random seed =====
    seed = 42  # can change to your preferred number
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # ===== Step 1: Argument parsing and environment setup =====
    try:
        logger.info("Parsing command line arguments...")
        args = parse_args()
        logger.info(f"Arguments parsed successfully: dataset={args.dataset}, split={args.split}, ratio={args.ratio}")
        logger.info(f"Target path: {args.target_path}")
        logger.info(f"Save path: {args.poison_save_pth}")
        # Display AMP mixed precision status
        if args.use_amp:
            logger.info("✓ AMP mixed precision training enabled (FP16) - can improve training speed")
        else:
            logger.info("○ AMP mixed precision training not enabled (default FP32)")
    except Exception as e:
        logger.info(f"Argument parsing failed: {e}")
        return

    # Set GPU device
    try:
        logger.info(f"Setting GPU device: {args.device}")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        logger.info("GPU device setup complete")
    except Exception as e:
        logger.info(f"GPU device setup failed: {e}")
        return

    # ===== Step 2: Output path processing =====
    try:
        logger.info("Processing output path...")
        # If no save path specified, use default path
        if args.poison_save_pth is None:
            ratio_str = f"{args.ratio:.4f}"
            subdir = f"{args.split}_{ratio_str}_{args.poison_data_last_name}"
            args.poison_save_pth = os.path.join(args.data_path, args.dataset, 'poisoned_image', subdir)
            logger.info(f"Using default save path: {args.poison_save_pth}")

        # If path does not exist, create it; if exists, use directly (allow repeated runs)
        os.makedirs(args.poison_save_pth, exist_ok=True)
        logger.info(f'Poisoned images will be saved to: {args.poison_save_pth}')
        logger.info(f'Attack parameters: iterations={args.iter_attack}, learning rate={args.lr_attack}')
    except Exception as e:
        logger.info(f"Output path processing failed: {e}")
        return

    # ===== Step 3: Prepare dataset configuration =====
    try:
        logger.info("Preparing dataset configuration...")
        dataset_info = {
            'dataset': args.dataset,      # dataset name
            'split': args.split,          # data split
            'ratio': args.ratio,          # data subset ratio
            'data_path': args.data_path   # data root directory
        }
        logger.info(f"Dataset configuration: {dataset_info}")
    except Exception as e:
        logger.info(f"Dataset configuration preparation failed: {e}")
        return

    # ===== Step 4: Model loading =====
    try:
        logger.info("Loading CLIP model...")
        logger.info(f"Using pretrained model: {args.from_pretrain}")
        clip_model, clip_processor = get_clip_model(args.from_pretrain)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Moving model to device: {device}")
        clip_model.to(device)
        # ===== Option A: statically convert vision_model to FP16, let 4090 truly run half-precision matmul =====
        if args.use_amp and torch.cuda.is_available():
            clip_model.vision_model = clip_model.vision_model.half()
            logger.info("✓ vision_model converted to FP16 (computation halved)")
        else:
            logger.info("vision_model kept FP32")
        # Freeze all CLIP parameters, not participating in gradient computation
        for param in clip_model.parameters():
            param.requires_grad = False
        logger.info("CLIP model loaded and frozen")
    except Exception as e:
        logger.info(f"CLIP model loading failed: {e}")
        return

    # ===== Step 5: Data loading =====
    try:
        logger.info("Loading images, masks and target data...")
        logger.info(f"Target path: {args.target_path}")
        logger.info(f"Image size: 224x224 (CLIP model standard input size)")

        images_base, images_target, masks, valid_image_ids, valid_image_rel_paths = load_image_tensors_with_masks(
            args.target_path, dataset_info, img_size=224  # CLIP model standard input size
        )

        # Transform mask region according to mask_mode
        logger.info(f"Original mask mode: {args.mask_mode}")
        if args.mask_mode != 'default':
            logger.info(f"Converting mask to mode={args.mask_mode} ...")
            masks = get_poisoned_mask(masks, mode=args.mask_mode)
            # Count coverage information of converted mask for troubleshooting
            coverage = masks[:, 0].mean().item()
            logger.info(f"Mask conversion complete, average coverage: {coverage:.4f}")
        else:
            logger.info("Using original masks in dataset, no conversion")

        logger.info(f"Successfully loaded {len(valid_image_ids)} valid images")
        logger.info(f"Image tensor shape: {images_base.shape}")
        logger.info(f"Mask tensor shape: {masks.shape}")

        # Create data loader for batch processing
        logger.info(f"Creating data loader, batch size: {args.batch_size}")
        dataset_pair = PairedImageDatasetWithMasks(images_base=images_base, images_target=images_target, masks=masks)
        dataloader_pair = torch.utils.data.DataLoader(dataset_pair, batch_size=args.batch_size, shuffle=False)
        logger.info(f"Data loader created, total batches: {len(dataloader_pair)}")

    except Exception as e:
        logger.info(f"Data loading failed: {e}")
        return

    # ===== Step 6: Execute PGD attack =====
    try:
        logger.info("Starting PGD attack...")
        # Poison each batch of data
        X_adv_list = []
        loss_attack_list = []
        total_batches = len(dataloader_pair)
        logger.info(f"Total {total_batches} batches to process")

        for i, (image_base, image_victim, mask) in enumerate(dataloader_pair):
            logger.info(f'Processing batch {i+1}/{total_batches}...')

            # Move data to GPU (if available)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Moving batch data to device: {device}")
            image_base, image_victim, mask = image_base.to(device), image_victim.to(device), mask.to(device)

            # Execute PGD attack based on critical regions
            logger.info(f"Executing PGD attack, iterations: {args.iter_attack}")
            if args.neg_contrast_weight > 0:
                logger.info(f"Negative contrastive loss enabled, weight: {args.neg_contrast_weight}")
            X_adv, loss_attack, loss_history = embedding_attack_Linf_with_mask(
                clip_model=clip_model,
                image_base=image_base,      # base image to be poisoned
                image_victim=image_victim,  # target image (embedding space target)
                mask=mask,                  # critical region mask
                emb_dist=cosine_similarity_loss,
                iters=args.iter_attack,
                lr=args.lr_attack/255,
                eps=args.eps,
                tv_weight=args.tv_weight,
                diff_aug=None,
                resume_X_adv=None,
                use_amp=args.use_amp,       # AMP mixed precision switch
                neg_contrast_weight=args.neg_contrast_weight  # negative contrastive loss weight
            )
            logger.info(f"Batch {i+1} attack complete, average loss: {loss_attack.mean().item():.4f}")

            # Plot loss curve
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Batch {i+1} Loss Curve')
            plt.legend()
            plt.grid(True)
            save_path = f'{args.data_path}/{args.dataset}/loss_curve_images/{args.split}_{args.ratio:.4f}_batch_{i+1}.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Loss curve saved to: {save_path}")

            # Move GPU results to CPU memory
            X_adv_list.append(X_adv.cpu())
            loss_attack_list.append(loss_attack.cpu())

        logger.info("All batches PGD attack complete")

    except Exception as e:
        logger.info(f"PGD attack execution failed: {e}")
        return

    # ===== Step 7: Result integration =====
    try:
        logger.info("Integrating attack results...")
        # Concatenate results of all batches into complete result tensors
        X_adv = torch.cat(X_adv_list, axis=0)          # [N, 3, 224, 224]
        loss_attack = torch.cat(loss_attack_list, dim=0)  # [N]

        logger.info(f"Result integration complete, total {len(X_adv)} poisoned images")
        logger.info(f"Average attack loss: {loss_attack.mean().item():.4f}")

    except Exception as e:
        logger.info(f"Result integration failed: {e}")
        return

    # ===== [New] Directly compute pixel distance (using in-memory tensors, avoid save/load errors) =====
    try:
        logger.info("Computing pixel distance (directly using post-attack tensors)...")
        # Directly use post-attack X_adv and original images_base to compute distance
        # images_base is already on GPU, need to move to CPU for computation
        images_base_cpu = images_base.cpu()
        X_adv_cpu = X_adv.cpu()

        # Compute L∞ distance (maximum absolute difference)
        pixel_dist = (images_base_cpu - X_adv_cpu).abs()
        max_pixel_dist = pixel_dist.max().item()
        mean_pixel_dist = pixel_dist.mean().item()

        logger.info(f"Pixel distance computed directly from tensors:")
        logger.info(f"  - Max pixel distance (L∞): {max_pixel_dist:.4f} (x255: {max_pixel_dist*255:.2f})")
        logger.info(f"  - Mean pixel distance: {mean_pixel_dist:.4f} (x255: {mean_pixel_dist*255:.2f})")
        logger.info(f"  - Constraint eps: {args.eps:.4f} (x255: {args.eps*255:.2f})")

        if max_pixel_dist > args.eps * 1.1:  # allow 10% numerical error
            logger.warning(f"Warning: max pixel distance exceeds eps constraint!")
        else:
            logger.info("✓ Pixel distance is within eps constraint")

    except Exception as e:
        logger.info(f"Computing pixel distance failed: {e}")

    # ===== Step 8: Save poisoned images =====
    try:
        logger.info("Saving poisoned images...")
        # Save optimized poisoned images to disk
        save_poison_data(
            images_to_save=X_adv,
            image_ids=valid_image_ids,
            image_rel_paths=valid_image_rel_paths,
            save_path=args.poison_save_pth,
            dataset_info=dataset_info,
            poison_data_last_name=args.poison_data_last_name,
            num_workers=None
        )
        logger.info("Poisoned images saved")
    except Exception as e:
        logger.info(f"Saving poisoned images failed: {e}")
        return

    # ===== Step 9: Attack effectiveness verification =====
    try:
        logger.info("Verifying attack effectiveness...")
        # Verify attack effectiveness on partial samples, compute embedding space distance and visual quality
        # Directly use in-memory tensors, avoid reloading images
        psnr, ssim = test_attack_efficacy(
            clip_model=clip_model,
            target_path=args.target_path,
            dataset_info=dataset_info,
            valid_image_ids=valid_image_ids,
            save_path=args.poison_save_pth,
            device=device,
            sample_num=50,  # only verify first 50 samples to save time
            emb_dist=cosine_similarity_loss,
            images_base_tensor=images_base,  # pass original image tensors
            X_adv_tensor=X_adv  # pass poisoned image tensors
        )
        logger.info("Attack effectiveness verification complete")

        # Output final statistics
        logger.info("\n=== Final attack statistics ===")
        logger.info(f"Number of processed images: {len(valid_image_ids)}")
        logger.info(f"Average PSNR: {psnr:.2f} dB (higher is better)")
        logger.info(f"Average SSIM: {ssim:.4f} (closer to 1 is better)")
        logger.info(f"Constraint strength: {args.eps*255:.1f}/255 ({args.eps*100:.1f}%)")
        logger.info(f"TV regularization weight: {args.tv_weight}")

    except Exception as e:
        logger.info(f"Attack effectiveness verification failed: {e}")

    logger.info(f'Poisoning attack complete! Results saved to {args.poison_save_pth}')


if __name__ == "__main__":
    # Initialize logging system
    setup_logging()
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nUser interrupted program execution")
    except Exception as e:
        logger.info(f"Uncaught exception during program execution: {e}")