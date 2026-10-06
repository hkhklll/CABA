#!/usr/bin/env python3
"""
Target Image Evaluation Tool

Features:
- Evaluate the effect of multiple target poisoning images
- Select target images from a specified directory by ratio
- Perform PGD optimization for each target image and evaluate the effect

Evaluation metrics:
- Embedding distance between original image and target image
- Embedding distance between poisoned image and target image
- Distance reduction ratio
- PSNR / SSIM

Output:
- Save information of good target images to good_target/good_target.txt
"""
import argparse
import os
import logging
import traceback
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPModel, CLIPProcessor
from torch.cuda.amp import autocast, GradScaler


# Configure logging system
def setup_logging(log_level=logging.INFO):
    """Configure logging system"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# Predefined resize transform
_RESIZE_224 = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
_TO_TENSOR = transforms.ToTensor()

# tqdm configuration - reduce refresh frequency
tqdm.monitor_interval = 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Target Image Evaluation Tool")
    parser.add_argument("--target_dir", default='./data/NUS-WIDE/target/person_9_train_set',
                        help='Target image directory path')
    parser.add_argument("--num_folds", type=int, default=2, help='Total number of folds for target images')
    parser.add_argument("--fold_index", type=int, default=0, help='Current fold index (starting from 0)')
    parser.add_argument("--iter_attack", type=int, default=1500, help='Number of PGD iterations')
    parser.add_argument("--lr_attack", type=float, default=0.7, help='PGD learning rate')
    parser.add_argument("--eps", type=float, default=4 / 255, help='L∞ constraint strength')
    parser.add_argument("--tv_weight", type=float, default=0.00004, help='TV regularization weight')
    parser.add_argument('--device', default='0', type=str, help='GPU device ID')
    parser.add_argument('--split', default='train', type=str, choices=['train', 'test', 'database'])
    parser.add_argument("--source_ratio", type=float, default=0.01,
                        help='Ratio of source images to use (0.01 means 1%)')
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--dataset', type=str, default='NUS-WIDE')
    parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_amp", type=bool, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=50)
    parser.add_argument("--output_dir", default='./data/NUS-WIDE/target/good_target', help='Output directory')
    args = parser.parse_args()
    return args


def load_image(image_path):
    """Load image"""
    return Image.open(image_path).convert('RGB')


def save_evaluate_poison_data(images_to_save, image_rel_paths, save_base_path, dataset_info, target_name, num_folds,
                              fold_index):
    """
    Save evaluated poisoned images to disk

    Args:
        images_to_save: poisoned image tensors
        image_rel_paths: list of image relative paths
        save_base_path: base save path
        dataset_info: dataset information dictionary
        target_name: target image name (used to distinguish different evaluations)
    """
    num_total = len(images_to_save)

    # Build save directory: evaluate_poisoned_image/{target_name}/
    replaced_dir = f'evaluate_poisoned_image/{num_folds}-{fold_index}/{target_name}'

    # Pre-collect all directories to create
    dirs_to_create = set()
    for i in range(num_total):
        split_idx = image_rel_paths[i].find('/')
        poison_filename = replaced_dir + image_rel_paths[i][split_idx:]
        img_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], poison_filename)
        dirs_to_create.add(os.path.dirname(img_pth))

    # Batch create directories
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    # Batch save images
    for i in range(num_total):
        split_idx = image_rel_paths[i].find('/')
        poison_filename = replaced_dir + image_rel_paths[i][split_idx:]
        img_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], poison_filename)
        save_image(images_to_save[i], img_pth)

    logger.info(f'Saved {num_total} evaluated poisoned images to {save_base_path}/{target_name}')


def load_evaluate_poison_data(image_rel_paths, save_base_path, dataset_info, num_folds, fold_index, target_name):
    """
    Load evaluated poisoned images from disk

    Args:
        image_rel_paths: list of image relative paths
        save_base_path: base save path
        dataset_info: dataset information dictionary
        target_name: target image name (used to distinguish different evaluations)

    Returns:
        loaded_images: loaded image tensors
    """
    num_total = len(image_rel_paths)
    loaded_images_list = []

    # Build load directory: evaluate_poisoned_image/{target_name}/
    replaced_dir = f'evaluate_poisoned_image/{num_folds}-{fold_index}/{target_name}'

    for i in range(num_total):
        split_idx = image_rel_paths[i].find('/')
        poison_filename = replaced_dir + image_rel_paths[i][split_idx:]
        img_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], poison_filename)

        try:
            img = load_image(img_pth)
            img_tensor = _TO_TENSOR(_RESIZE_224(img))
            loaded_images_list.append(img_tensor)
        except Exception as e:
            logger.warning(f"Failed to load poisoned image {img_pth}: {e}")
            # If loading fails, return original image
            split_idx = image_rel_paths[i].find('/')
            base_filename = image_rel_paths[i][split_idx:]
            base_img_pth = os.path.join(dataset_info['data_path'], dataset_info['dataset'], base_filename)
            img = load_image(base_img_pth)
            img_tensor = _TO_TENSOR(_RESIZE_224(img))
            loaded_images_list.append(img_tensor)

    loaded_images = torch.stack(loaded_images_list)
    return loaded_images
    return Image.open(image_path).convert('RGB')


# CLIP model normalization
normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711)
)


def get_clip_model(from_pretrain):
    """Load CLIP model"""
    model = CLIPModel.from_pretrained(from_pretrain)
    processor = CLIPProcessor.from_pretrained(from_pretrain)
    model.eval()
    return model, processor


def total_variation_loss(x, mask):
    """Compute total variation loss"""
    x_masked = x * mask
    tv_h = torch.abs(x_masked[:, :, :, :-1] - x_masked[:, :, :, 1:])
    tv_v = torch.abs(x_masked[:, :, :-1, :] - x_masked[:, :, 1:, :])
    mask_h = mask[:, :, :, :-1]
    mask_v = mask[:, :, :-1, :]
    tv_loss = (tv_h * mask_h).sum() + (tv_v * mask_v).sum()
    return tv_loss / x.size(0)


def L2_norm(a, b):
    """Compute L2 distance"""
    assert a.size(0) == b.size(0)
    bs = a.size(0)
    return (a - b).view(bs, -1).norm(p=2, dim=1)


def embedding_attack_Linf_with_mask(clip_model, image_base, image_victim, mask,
                                    iters=100, lr=1 / 255, eps=8 / 255, tv_weight=0.001,
                                    use_amp=True, early_stop_patience=50, target_embedding=None):
    """
    PGD attack algorithm
    Returns: (X_adv_best, final_loss, early_stop_info)
    """
    device = image_base.device
    bs = image_base.size(0)

    # Early stopping statistics
    early_stop_count = 0
    early_stop_iter = -1

    # Precompute target embedding
    with torch.no_grad():
        if target_embedding is None:
            embedding_targets = clip_model.vision_model(normalize(image_victim)).pooler_output
        else:
            embedding_targets = target_embedding.expand(bs, -1).to(device)

    # Initialize adversarial examples
    X_adv = image_base.clone().detach() + (torch.rand(*image_base.shape, device=device) * 2 * eps - eps)
    X_adv = X_adv.clamp(0, 1).requires_grad_(True)

    optimizer = optim.Adam([X_adv], lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=lr * 0.05)

    loss_best = torch.full((bs,), 1e8, device=device)
    X_adv_best = X_adv.clone().detach()

    # Early stopping mechanism
    no_improve_count = torch.zeros(bs, device=device, dtype=torch.int32)
    all_converged = torch.zeros(bs, device=device, dtype=torch.bool)

    scaler = GradScaler() if use_amp else None
    mask_complement = (mask < 0.5).float()

    print_freq = max(int(iters * 0.05), 1)
    check_best_freq = max(int(iters * 0.2), 1)

    for i in range(iters):
        if all_converged.all():
            early_stop_count = all_converged.sum().item()
            early_stop_iter = i
            logger.debug(f'Early stop: {early_stop_count}/{bs} samples converged at iteration {i}')
            break

        # Forward pass
        if use_amp and scaler is not None:
            with autocast():
                X_adv_input = normalize(X_adv)
                embedding_adv = clip_model.vision_model(X_adv_input).pooler_output
                embedding_loss = L2_norm(embedding_adv, embedding_targets)
                tv_loss = total_variation_loss(X_adv, mask)
                loss = embedding_loss + tv_weight * tv_loss
        else:
            X_adv_input = normalize(X_adv)
            embedding_adv = clip_model.vision_model(X_adv_input).pooler_output
            embedding_loss = L2_norm(embedding_adv, embedding_targets)
            tv_loss = total_variation_loss(X_adv, mask)
            loss = embedding_loss + tv_weight * tv_loss

        # Check best result
        if i % check_best_freq == 0:
            improved = (loss < loss_best - 1e-4)
            loss_best = torch.where(improved, loss, loss_best)
            indices = torch.where(improved)[0]
            if indices.numel() > 0:
                X_adv_best[indices] = X_adv[indices].clone().detach()

            no_improve_count += (~improved).int()
            all_converged = (no_improve_count >= early_stop_patience)

        # Backward pass
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss.mean()).backward()
        else:
            loss.mean().backward()

        if X_adv.grad is not None:
            X_adv.grad = X_adv.grad * mask
            X_adv.grad = torch.sign(X_adv.grad)

        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        # Project to constraint space
        X_adv.data = mask * torch.clamp(X_adv, image_base - eps, image_base + eps) + mask_complement * image_base
        X_adv.data = X_adv.data.clamp(0, 1)
        X_adv.grad = None

        if i % print_freq == 0 and i > 0:
            with torch.no_grad():
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f'Iter {i}: loss={loss.mean().item():.4f}, lr={current_lr * 255:.4f}, converged={all_converged.sum().item()}/{bs}')

        if torch.isnan(loss).any():
            logger.warning(f'Encountered NaN loss at iteration {i}')
            break

    # Final evaluation
    with torch.no_grad():
        X_adv_input = normalize(X_adv_best)
        embedding_final = clip_model.vision_model(X_adv_input).pooler_output
        final_loss = L2_norm(embedding_final, embedding_targets)

    return X_adv_best, final_loss.detach(), X_adv_best, {'early_stop_count': early_stop_count,
                                                         'early_stop_iter': early_stop_iter}


def calculate_psnr_ssim(original_images, poisoned_images):
    """Calculate PSNR and SSIM"""
    try:
        from skimage.metrics import structural_similarity as ssim

        orig_np = original_images.cpu().numpy()
        poison_np = poisoned_images.cpu().numpy()

        # MSE
        mse_values = np.mean((orig_np - poison_np) ** 2, axis=(1, 2, 3))
        mse_values = np.where(mse_values == 0, 1e-10, mse_values)

        # PSNR
        psnr_values = 20 * np.log10(1.0 / np.sqrt(mse_values))

        # SSIM
        ssim_values = np.zeros(len(original_images))
        for c in range(3):
            ssim_values += np.array([
                ssim(orig_np[i, c], poison_np[i, c], data_range=1.0)
                for i in range(len(original_images))
            ])
        ssim_values /= 3

        return psnr_values.mean(), ssim_values.mean()

    except ImportError:
        logger.warning("Cannot compute SSIM")
        orig_np = original_images.cpu().numpy()
        poison_np = poisoned_images.cpu().numpy()
        mse_values = np.mean((orig_np - poison_np) ** 2, axis=(1, 2, 3))
        mse_values = np.where(mse_values == 0, 1e-10, mse_values)
        psnr_values = 20 * np.log10(1.0 / np.sqrt(mse_values))
        return psnr_values.mean(), 0.0


def load_source_images(dataset_info, ratio=0.01):
    """Load source images and corresponding key region masks for evaluation (consistent with poison_image_regions.py)"""
    img_list_file = os.path.join(dataset_info['data_path'], dataset_info['dataset'],
                                 f'cm_{dataset_info["split"]}_imgs.txt')

    if not os.path.exists(img_list_file):
        raise ValueError(f"Image list file does not exist: {img_list_file}")

    with open(img_list_file, 'r') as f:
        image_rel_paths = [line.strip() for line in f if line.strip()]

    subset_size = int(len(image_rel_paths) * ratio)
    image_rel_paths = image_rel_paths[:subset_size]

    logger.info(f"Loading {len(image_rel_paths)} source images")

    # Pre-allocate list to collect valid data
    valid_data = []
    subdir = dataset_info['split']

    # First pass: collect valid image-mask pairs
    for image_rel_path in image_rel_paths:
        image_id = os.path.splitext(os.path.basename(image_rel_path))[0]
        image_path = os.path.join(dataset_info['data_path'], dataset_info['dataset'], image_rel_path)
        # Build mask path: replace 'images' with 'masks_image/{subdir}'
        mask_rel_path = image_rel_path.replace('images', f'masks_image/{subdir}', 1)
        mask_path = os.path.join(dataset_info['data_path'], dataset_info['dataset'], mask_rel_path)
        mask_path = mask_path.replace('jpg', 'png')
        # Check if files exist
        if os.path.exists(image_path) and os.path.exists(mask_path):
            valid_data.append((image_path, mask_path, image_id, image_rel_path))

    logger.info(f"Found {len(valid_data)} valid image-mask pairs")

    if len(valid_data) == 0:
        raise ValueError("No valid image-mask pairs found")

    # Pre-allocate numpy arrays
    num_images = len(valid_data)
    img_size = 224
    images_base_np = np.zeros((num_images, 3, img_size, img_size), dtype=np.float32)
    masks_np = np.zeros((num_images, 3, img_size, img_size), dtype=np.float32)

    # Collect IDs and paths of valid images
    valid_image_ids = []
    valid_image_rel_paths = []

    # Second pass: batch load images and masks
    for idx, (image_path, mask_path, image_id, image_rel_path) in enumerate(
            tqdm(valid_data, desc="Loading images", mininterval=5.0)):
        try:
            # Load image and mask, convert and resize
            base_tensor = _TO_TENSOR(_RESIZE_224(load_image(image_path)))
            mask_array = np.array(_RESIZE_224(load_image(mask_path)))

            # Process mask: consistent with poison_image_regions.py
            if len(mask_array.shape) == 3:
                mask_gray = np.mean(mask_array, axis=2)
            else:
                mask_gray = mask_array
            mask_binary = (mask_gray / 255.0).astype(np.float32)
            mask_binary = np.stack([mask_binary] * 3, axis=2)

            # Store into pre-allocated arrays
            images_base_np[idx] = base_tensor.numpy()
            masks_np[idx] = mask_binary.transpose(2, 0, 1)

            valid_image_ids.append(image_id)
            valid_image_rel_paths.append(image_rel_path)

        except Exception as e:
            logger.error(f"Error loading image {image_id}: {e}")
            continue

    # Convert to torch tensors (consistent with poison_image_regions.py, outside for loop)
    images_base = torch.from_numpy(images_base_np)
    masks = torch.from_numpy(masks_np)

    logger.info(f"Successfully loaded {len(images_base)} source images")
    logger.info(f"Mask tensor shape: {masks.shape}")

    return images_base, masks, valid_image_ids, valid_image_rel_paths


def evaluate_target_image(clip_model, target_tensor, source_images, masks, valid_image_rel_paths, dataset_info, args):
    """
    Evaluate the effect of a single target image (save to file first, then load from file for validation)

    Args:
        clip_model: CLIP model
        target_tensor: target image tensor
        source_images: source image tensors
        masks: key region mask tensors (consistent with poison_image_regions.py)
        valid_image_rel_paths: list of valid image relative paths
        dataset_info: dataset information dictionary
        args: command line arguments
    """
    device = source_images.device
    bs = source_images.size(0)

    # Get target image name (remove extension)
    target_name = os.path.splitext(args.current_target_name)[0]

    # Expand target image to the same number
    target_expanded = target_tensor.expand(bs, -1, -1, -1)

    # Use the passed region masks (consistent with poison_image_regions.py)
    mask = masks

    # Precompute target embedding
    with torch.no_grad():
        target_embedding = clip_model.vision_model(normalize(target_expanded)).pooler_output

    # PGD attack
    X_adv, final_loss, _, early_stop_info = embedding_attack_Linf_with_mask(
        clip_model=clip_model,
        image_base=source_images,
        image_victim=target_expanded,
        mask=mask,
        iters=args.iter_attack,
        lr=args.lr_attack / 255,
        eps=args.eps,
        tv_weight=args.tv_weight,
        use_amp=args.use_amp,
        early_stop_patience=args.early_stop_patience,
        target_embedding=target_embedding
    )

    # ===== Save poisoned images to file first =====
    save_base_path = os.path.join(dataset_info['data_path'], dataset_info['dataset'])
    save_evaluate_poison_data(
        images_to_save=X_adv,
        image_rel_paths=valid_image_rel_paths,
        save_base_path=save_base_path,
        dataset_info=dataset_info,
        target_name=target_name,
        num_folds=args.num_folds,
        fold_index=args.fold_index
    )

    # ===== Load poisoned images from file =====
    X_adv_loaded = load_evaluate_poison_data(
        image_rel_paths=valid_image_rel_paths,
        save_base_path=save_base_path,
        dataset_info=dataset_info,
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        target_name=target_name
    ).to(device)

    # Compute embedding distance between original and target
    with torch.no_grad():
        source_normalized = normalize(source_images)
        target_normalized = normalize(target_expanded)
        emb_source = clip_model.vision_model(source_normalized).pooler_output
        emb_target = clip_model.vision_model(target_normalized).pooler_output

        dist_source_target = L2_norm(emb_source, emb_target)

        # Compute embedding distance using poisoned images loaded from file
        poison_normalized = normalize(X_adv_loaded)
        emb_poison = clip_model.vision_model(poison_normalized).pooler_output
        dist_poison_target = L2_norm(emb_poison, emb_target)

    # Compute PSNR/SSIM (use original X_adv and source_images, as this is visual quality)
    psnr, ssim = calculate_psnr_ssim(source_images, X_adv)

    # Compute distance reduction ratio (using distances loaded from file)
    dist_reduction_ratio = (dist_source_target - dist_poison_target) / (dist_source_target + 1e-8)

    return {
        'dist_source_target': dist_source_target.mean().item(),
        'dist_poison_target': dist_poison_target.mean().item(),
        'dist_reduction_ratio': dist_reduction_ratio.mean().item(),
        'psnr': psnr,
        'ssim': ssim,
        'early_stop_count': early_stop_info.get('early_stop_count', 0),
        'early_stop_iter': early_stop_info.get('early_stop_iter', -1)
    }


def get_target_images(target_dir, num_folds=1, fold_index=0):
    """Get target image list (supports folds)

    Args:
        target_dir: target image directory
        num_folds: total number of folds
        fold_index: current fold index (starting from 0)
    """
    # Supported image formats
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    all_images = []
    for fname in os.listdir(target_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            all_images.append(os.path.join(target_dir, fname))

    all_images.sort()

    # If only one fold, return all directly
    if num_folds <= 1:
        logger.info(f"Total {len(all_images)} target images")
        return all_images

    # Fold processing
    fold_size = len(all_images) // num_folds
    start_idx = fold_index * fold_size

    # The last fold contains all remaining images
    if fold_index == num_folds - 1:
        end_idx = len(all_images)
    else:
        end_idx = start_idx + fold_size

    fold_images = all_images[start_idx:end_idx]

    logger.info(
        f"Total {len(all_images)} target images, fold {fold_index + 1}/{num_folds}: taking indices {start_idx} to {end_idx - 1}, {len(fold_images)} images")

    return fold_images


def main():
    """Main function"""
    # Initialize logging system
    setup_logging()

    logger.info("=== Starting target image evaluation ===")

    # Parse arguments
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Get target image list (supports folds)
    target_images = get_target_images(args.target_dir, args.num_folds, args.fold_index)
    logger.info(f"Target image directory: {args.target_dir}")
    logger.info(f"Fold: {args.fold_index + 1} of {args.num_folds}, total {len(target_images)} images")

    # Load CLIP model
    logger.info("Loading CLIP model...")
    clip_model, _ = get_clip_model(args.from_pretrain)
    clip_model.to(device)
    clip_model.eval()
    logger.info("CLIP model loaded")

    # Prepare dataset information
    dataset_info = {
        'dataset': args.dataset,
        'split': args.split,
        'data_path': args.data_path
    }

    # Load source images and masks
    logger.info("Loading source images and masks...")
    source_images, masks, valid_image_ids, valid_image_rel_paths = load_source_images(dataset_info,
                                                                                      ratio=args.source_ratio)
    source_images = source_images.to(device)
    masks = masks.to(device)

    # Evaluate each target image
    results = []

    for target_path in tqdm(target_images, desc="Evaluating target images", mininterval=5.0):
        try:
            target_name = os.path.basename(target_path)
            logger.info(f"Evaluating target: {target_name}")

            # Load target image
            target_img = load_image(target_path)
            target_tensor = _TO_TENSOR(_RESIZE_224(target_img)).to(device)

            # Set current target name (for saving poisoned images)
            args.current_target_name = target_name

            # Evaluate (save to file first, then load from file for validation)
            metrics = evaluate_target_image(
                clip_model, target_tensor, source_images, masks,
                valid_image_rel_paths, dataset_info, args
            )
            metrics['target_name'] = target_name

            logger.info(f"  Original-target distance: {metrics['dist_source_target']:.4f}")
            logger.info(f"  Poisoned-target distance: {metrics['dist_poison_target']:.4f}")
            logger.info(f"  Distance reduction ratio: {metrics['dist_reduction_ratio']:.4f}")
            logger.info(f"  PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}")

            # Early stopping information
            if metrics['early_stop_iter'] > 0:
                logger.info(f"  [Early stop] Converged at iteration {metrics['early_stop_iter']}")

            results.append(metrics)

        except Exception as e:
            logger.error(f"Failed to evaluate target {target_path}: {e}")
            traceback.print_exc()
            continue

    # Filter results with positive distance reduction ratio
    good_results = [r for r in results if r['dist_reduction_ratio'] > 0]

    logger.info(f"\nEvaluation complete! Evaluated {len(results)} target images")
    logger.info(f"Number of targets with positive distance reduction ratio: {len(good_results)}")

    # Save results (including fold index)
    output_file = os.path.join(output_dir, f'good_target_{args.num_folds}-{args.fold_index}.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Target image evaluation results\n")
        f.write(f"# Number of evaluated targets: {len(results)}\n")
        f.write(f"# Number of valid targets (distance reduction ratio > 0): {len(good_results)}\n")
        f.write(
            f"# Parameters: iter={args.iter_attack}, lr={args.lr_attack}, eps={args.eps}, tv_weight={args.tv_weight}\n")
        f.write(f"\n")

        f.write(
            f"{'Target Name':<30} {'Orig-Target Dist':<15} {'Poison-Target Dist':<15} {'Dist Reduction':<15} {'PSNR':<10} {'SSIM':<10}\n")
        f.write("-" * 95 + "\n")

        good_results.sort(key=lambda r: r['dist_reduction_ratio'], reverse=True)

        for r in good_results:
            f.write(f"{r['target_name']:<30} {r['dist_source_target']:<15.4f} {r['dist_poison_target']:<15.4f} "
                    f"{r['dist_reduction_ratio']:<15.4f} {r['psnr']:<10.2f} {r['ssim']:<10.4f}\n")

    logger.info(f"Results saved to: {output_file}")

    # Print statistics
    if good_results:
        reduction_ratios = [r['dist_reduction_ratio'] for r in good_results]
        psnrs = [r['psnr'] for r in good_results]
        ssims = [r['ssim'] for r in good_results]

        logger.info("\n=== Valid target statistics ===")
        logger.info(
            f"Distance reduction ratio: mean={np.mean(reduction_ratios):.4f}, max={np.max(reduction_ratios):.4f}")
        logger.info(f"PSNR: mean={np.mean(psnrs):.2f}, max={np.max(psnrs):.2f}")
        logger.info(f"SSIM: mean={np.mean(ssims):.4f}, max={np.max(ssims):.4f}")

    # Early stopping statistics
    early_stop_count_total = sum(1 for r in results if r.get('early_stop_iter', -1) > 0)
    if early_stop_count_total > 0:
        early_stop_iters = [r['early_stop_iter'] for r in results if r.get('early_stop_iter', -1) > 0]
        logger.info(f"\n=== Early stopping statistics ===")
        logger.info(f"Number of early-stopped targets: {early_stop_count_total}/{len(results)}")
        logger.info(
            f"Early stop iterations: mean={np.mean(early_stop_iters):.1f}, min={np.min(early_stop_iters)}, max={np.max(early_stop_iters)}")

    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nUser interrupted program execution")
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        traceback.print_exc()