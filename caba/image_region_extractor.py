# Use GroundingDINO model to extract modality-invariant components from images based on text descriptions
# Based on zero-shot object detection and text-image alignment
#
import os
import re
import argparse
import pickle
import torch
import json
from datetime import datetime
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from dataset.dataset import CrossModalDataset
from dataset.dataset import get_dataset_filename
from utils.utils import check_path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


class ImageRegionExtractor:
    """
    Image Region Extractor

    Core functions:
    1. Use GroundingDINO to detect key objects in images based on text descriptions
    2. Extract keywords from text as detection prompts
    3. Identify critical regions based on similarity analysis
    4. Generate modality-invariant component regions

    Theoretical basis:
    - Zero-shot object detection
    - Text-image alignment
    - Open-vocabulary visual understanding
    """

    def __init__(self, args):
        """
        Initialize Image Region Extractor

        Args:
        - args: command line arguments, including model configuration, dataset paths, etc.
        """
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_text_len = args.max_text_len
        # Data subset ratio (used to locate bbox files saved by external interface)
        self.ratio = getattr(args, 'ratio', 0.01)
        # Input file path (used to specify processing only certain files)
        self.input_file = getattr(args, 'input_file', None)

        # Generate timestamp for this run (used for result file naming)
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        print(f"Using device: {self.device}")
        print(f"Max text length: {self.max_text_len}")

        # Load GroundingDINO model
        # No longer load GroundingDINO model; preliminary anchor boxes are obtained by external interface and saved to data/{dataset}/bbox/
        # Still load CLIP model (for similarity computation)
        self.clip_model, self.clip_processor = self._load_clip_model()
        self.clip_model.to(self.device)
        print("CLIP model loaded")

        # Load dataset
        self.dataset, self.selected_image_names = self._load_dataset()
        print(f"Dataset loaded: {args.dataset} - {args.split}")

        # Lazily load saved keyword file (load when actually needed)
        self.saved_text_words = None

        # Stop words list (for keyword extraction)
        self.stop_words = self._get_stop_words()

        # GroundingDINO detection parameters
        self.box_threshold = getattr(args, 'box_threshold', 0.3)  # bounding box confidence threshold

        # Similarity analysis parameters
        self.cross_modal_weight = getattr(args, 'cross_modal_weight', 0.5)  # cross-modal similarity weight

        print(f"GroundingDINO detection parameters:")
        print(f"  box_threshold: {self.box_threshold}")
        print(f"Similarity analysis parameters:")
        print(f"  cross-modal similarity weight: {self.cross_modal_weight:.2f}")
        print(f"  image similarity weight: {(1 - self.cross_modal_weight):.2f}")

        # ---- The following is initialization for acceleration optimization ----
        # Pre-parse image_name mapping list (avoid repeatedly accessing Subset.indices + imgs attributes in main loop)
        self._img_id_to_name = [None] * len(self.dataset)
        self._img_id_to_path = [None] * len(self.dataset)
        self._prepare_img_id_mapping()

        # Preload bbox JSON (avoid repeated json.load and dict.get string fallback in main loop)
        self._saved_bboxes = None
        self._load_bboxes_cache()

        # Async disk write thread pool (lazily initialized when first needed)
        self._async_pool = None

    def _prepare_img_id_mapping(self):
        """Pre-parse each sample's image_name / image_path into lists to avoid repeated lookups in main loop"""
        original_dataset = self.dataset
        if isinstance(self.dataset, torch.utils.data.Subset):
            original_dataset = self.dataset.dataset

        for idx in range(len(self.dataset)):
            dataset_idx = idx
            if isinstance(self.dataset, torch.utils.data.Subset):
                dataset_idx = self.dataset.indices[idx]
            try:
                img_path = original_dataset.imgs[dataset_idx]
                self._img_id_to_path[idx] = img_path
                self._img_id_to_name[idx] = os.path.basename(img_path)
            except Exception:
                self._img_id_to_path[idx] = None
                self._img_id_to_name[idx] = None

    def _load_bboxes_cache(self):
        """Load bbox JSON once during init to avoid re-reading for each sample"""
        input_file = getattr(self.args, 'input_file', None)
        if input_file and input_file.strip():
            bbox_file = os.path.join(
                self.args.data_path, self.args.dataset, "bbox",
                "NUS-WIDE_selected_train_bbox_r=0.0100.json"
            )
        else:
            ratio_str = f"{self.ratio:.4f}"
            bbox_file = os.path.join(
                self.args.data_path, self.args.dataset, "bbox",
                f"{self.args.dataset}_{self.args.split}_bbox_r={ratio_str}.json"
            )

        if not os.path.exists(bbox_file):
            print(f"  Warning: bbox file does not exist {bbox_file}, will process dynamically at runtime")
            return

        try:
            raw = json.load(open(bbox_file, "r", encoding="utf-8"))
            # Build dict index from file_name -> objects
            self._saved_bboxes = {os.path.basename(k): v for k, v in raw.items()}
            print(f"  bbox JSON preloaded, total {len(self._saved_bboxes)} records")
        except Exception as e:
            print(f"  Failed to preload bbox file: {e}, will process dynamically at runtime")
            self._saved_bboxes = None

    def _load_clip_model(self):
        """
        Load CLIP model and processor (for similarity computation)

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

        Supports two data selection modes:
        1. Ratio mode: use ratio parameter to sequentially select the first N samples from the full dataset
        2. File mode: use input_file parameter, process only images contained in the specified file

        Returns:
        - dataset: torch.utils.data.Subset or CrossModalDataset
        - selected_image_names: set of selected image names (used for matching in file mode)
        """
        transform = transforms.Compose([lambda x: np.array(x)])
        img_name, text_name, label_name = get_dataset_filename(self.args.split)
        full_dataset = CrossModalDataset(
            os.path.join(self.args.data_path, self.args.dataset),
            img_name, text_name, label_name, transform
        )

        # Determine data selection mode
        input_file = getattr(self.args, 'input_file', None)
        ratio = getattr(self.args, 'ratio', 0.01)

        if input_file is not None and input_file.strip():
            # Mode 2: file selection
            print(f"Using file mode: {input_file}")
            selected_image_names = self._load_selected_images(input_file)

            # Create index mapping: only include images in the specified file
            selected_indices = []
            for idx in range(len(full_dataset)):
                img_path = full_dataset.imgs[idx]
                img_basename = os.path.basename(img_path)
                if img_basename in selected_image_names:
                    selected_indices.append(idx)

            dataset = torch.utils.data.Subset(full_dataset, selected_indices)
            print(
                f"Full dataset size: {len(full_dataset)}, specified in file: {len(selected_image_names)}, matched: {len(dataset)}")

        else:
            # Mode 1: ratio selection
            print(f"Using ratio mode: ratio={ratio}")
            selected_image_names = None

            if ratio >= 1.0:
                # Use all data
                dataset = full_dataset
                print(f"Full dataset size: {len(dataset)}")
            else:
                # Use only the first ratio portion of data
                subset_size = int(len(full_dataset) * ratio)
                dataset = torch.utils.data.Subset(full_dataset, range(subset_size))
                print(f"Full dataset size: {len(full_dataset)}, used data: {len(dataset)} (ratio={ratio})")

        return dataset, selected_image_names

    def _load_selected_images(self, input_file):
        """
        Read the list of image names to process from a file

        Supports two formats:
        1. One image path or image name per line
        2. Each line is "path score" (score is ignored)

        Args:
        - input_file: file path

        Returns:
        - image_names: set[str] set of image names
        """
        image_names = set()

        # If relative path, try relative to current working directory
        file_path = input_file
        if not os.path.isabs(file_path) and not os.path.exists(file_path):
            # Try relative to parent directory of current working directory
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)

        print(f"Reading input file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse filename (take only the last part/filename of the path)
                # Handle two formats:
                # 1. "images/xxx.jpg"
                # 2. "images/xxx.jpg 0.123456"
                parts = line.split()
                filename = parts[0]

                # If path contains directories, take the last filename
                if '/' in filename:
                    basename = os.path.basename(filename)
                else:
                    basename = filename

                image_names.add(basename)

        print(f"Read {len(image_names)} image names from file")
        return image_names

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
            "don", "should", "now"
        }

    def _get_dataset_img_info(self, image_id):
        """
        Return (img_path, img_basename) or (None, None).
        Use precomputed mapping to avoid repeatedly looking up Subset.indices + imgs.
        """
        if image_id is None:
            return None, None
        try:
            idx = int(image_id)
            if 0 <= idx < len(self._img_id_to_path):
                return self._img_id_to_path[idx], self._img_id_to_name[idx]
        except (ValueError, TypeError):
            pass
        return None, None

    def _center_region(self, img_width: float, img_height: float):
        """
        When identify_critical_image_regions detects no regions, no valid regions after initial filtering, or selected_regions is empty,
        return the center region of the entire image.
        Returns a single-element list representing the center region (bbox width and height are half of the image).
        Format is consistent with other regions (includes "bbox", "score", "label", "area").
        """
        w = img_width / 2.0
        h = img_height / 2.0
        x_min = (img_width - w) / 2.0
        y_min = (img_height - h) / 2.0
        x_max = x_min + w
        y_max = y_min + h
        center_region = [{
            "bbox": np.array([x_min, y_min, x_max, y_max]),
            "score": 1.0,
            "label": "center_region",
            "area": int(w * h)
        }]
        return center_region

    def _get_empty_detections(self):
        """Return empty detection results"""
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'labels': []
        }

    def detect_regions_with_grounding_dino(self, image_id=None):
        """
        Detect key regions in the image using preloaded bbox JSON.

        Args:
        - image_id: image ID (used to look up keywords from saved keyword file)

        Returns:
        - detections: detection result dict, containing boxes, labels, scores
        """
        # Directly query from preloaded cache (O(1), no json.load / hasattr overhead)
        image_name = None
        if image_id is not None:
            _, image_name = self._get_dataset_img_info(image_id)

        objects = []
        if image_name and self._saved_bboxes:
            objects = self._saved_bboxes.get(image_name, [])

        if not objects:
            return self._get_empty_detections()

        # Build detections format
        boxes = []
        scores = []
        labels = []
        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) >= 4:
                boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                scores.append(float(obj.get("score")))
                labels.append(obj.get("category"))

        if len(boxes) == 0:
            return self._get_empty_detections()

        detections = {
            "boxes": np.array(boxes),
            "scores": np.array(scores),
            "labels": labels
        }
        return detections

    def _compute_cross_modal_similarity(self, text, image):
        """
        Compute text-image cross-modal similarity

        Args:
        - text: text string
        - image: PIL Image

        Returns:
        - similarity: cosine similarity value
        - image_embeds: image embeddings (for subsequent computation)
        """
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

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarity = F.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds
            ).item()
            image_embeds = outputs.image_embeds

        return similarity, image_embeds

    def _encode_images_batch(self, images):
        """
        Batch encode a group of images (used to compute CLIP image_embeds for multiple images at once).

        Args:
        - images: List[PIL.Image]

        Returns:
        - image_embeds: Tensor (N, D), already normalized
        """
        inputs = self.clip_processor(images=images, return_tensors="pt")
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            image_embeds = F.normalize(outputs, p=2, dim=1)
        return image_embeds

    def _encode_image_text_batch(self, images, text):
        """
        Batch compute CLIP similarity for image-text pairs (single text vs multiple images).

        Args:
        - images: List[PIL.Image]
        - text:  str, single text description

        Returns:
        - image_embeds: Tensor (N, D)
        - text_embeds:   Tensor (1, D), broadcast aligned with images
        - similarities:  Tensor (N,), cosine similarity
        """
        inputs = self.clip_processor(
            text=text,
            images=images,
            return_tensors="pt",
            max_length=self.max_text_len,
            padding='longest',
            truncation='longest_first'
        )
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            similarities = F.cosine_similarity(image_embeds, text_embeds, dim=1)
        return image_embeds, text_embeds, similarities

    def identify_critical_image_regions(self, detections, image, text=None, image_id=None):
        """
        Identify critical image regions - based on GroundingDINO detection results and similarity analysis

        Args:
        - detections: GroundingDINO detection result dict, containing boxes, scores, labels
        - image: original image
        - text: original text description (used for similarity computation)
        - image_id: image ID (used to get image name)

        Returns:
        - critical_regions: list of critical regions
        """
        # Get image name for printing
        _, image_name = self._get_dataset_img_info(image_id)
        image_name = image_name or f"image_{image_id}"
        print(f"  Starting to identify critical regions... (image: {image_name})")
        # Get image dimensions
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
            img_array = np.array(image)
        # elif isinstance(image, np.ndarray):
        #     img_height, img_width = image.shape[:2]
        #     img_array = image.copy()
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")

        boxes = detections.get('boxes', np.array([]))
        scores = detections.get('scores', np.array([]))
        labels = detections.get('labels', [])

        if len(boxes) == 0:
            print("  Warning: no regions detected, returning center region of image")
            return self._center_region(img_width, img_height)

        # Area thresholds
        min_bbox_area = img_width * img_height * 0.001  # minimum bounding box 0.1%
        max_bbox_area = img_width * img_height * 0.5  # maximum bounding box 50%

        # Confidence threshold
        confidence_threshold = self.box_threshold

        # Step 1: parse all boxes and perform initial filtering
        valid_regions = []

        for i, box in enumerate(boxes):
            if len(box) != 4:
                continue

            # Filter out boxes with too low confidence
            score = float(scores[i]) if i < len(scores) else 0.0
            if score < confidence_threshold:
                continue

            label = labels[i] if i < len(labels) else "grounding_dino_region"

            # GroundingDINO 1.6pro returns pixel coordinates
            box_values = np.array(box).flatten()

            if len(box_values) == 4:
                x_min, y_min, x_max, y_max = box_values
            else:
                continue

            # Ensure coordinates are within image boundaries
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            x_max = max(x_min, min(x_max, img_width))
            y_max = max(y_min, min(y_max, img_height))

            bbox_area = (x_max - x_min) * (y_max - y_min)

            # Filter out regions that are too small or too large
            if bbox_area < min_bbox_area:
                print(f"Region area is {bbox_area}, smaller than minimum threshold {min_bbox_area}, filtered out")
                continue
            if bbox_area > max_bbox_area:
                print(f"Region area is {bbox_area}, larger than maximum threshold {max_bbox_area}, filtered out")
                continue

            valid_regions.append({
                "bbox": np.array([x_min, y_min, x_max, y_max]),
                "score": score,
                "label": label,
                "area": int(bbox_area)
            })

        print(
            f"Before initial filtering there were {len(boxes)} regions, after initial filtering {len(valid_regions)} regions remain (confidence >= {confidence_threshold:.3f})")

        if len(valid_regions) == 0:
            print("  Warning: no valid regions after initial filtering, returning center region of image")
            return self._center_region(img_width, img_height)

        # Step 2: if text is provided, compute similarity-based sensitivity scores
        if text is not None:
            print("  Starting to compute similarity-based sensitivity scores...")
            img_pil = Image.fromarray(img_array) if not isinstance(image, Image.Image) else image

            # ---- batch CLIP forward (merge original image + all region masked images) ----
            # Pre-build masked image list for all regions
            masked_img_list = []
            try:
                for region in valid_regions:
                    x_min, y_min, x_max, y_max = region["bbox"]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    masked = img_array.copy().astype(np.float32)
                    masked[y_min:y_max, x_min:x_max] = 1
                    masked = np.clip(masked, 0, 255)
                    masked_img_list.append(Image.fromarray(masked.astype(np.uint8)))
            except Exception as e:
                print(f"  Warning: error generating region masked images: {str(e)}, skipping similarity computation")
                text = None

        if text is not None:
            # One-time batch forward: original image + all masked images
            try:
                all_images = [img_pil] + masked_img_list
                image_embeds_all, _, cos_sims_all = self._encode_image_text_batch(all_images, text)
                cos_sim_original = cos_sims_all[0].item()  # (1,)
                cos_sims_masked = cos_sims_all[1:]  # (N,)

                # Vectorized: original image embedding vs all masked embeddings
                image_embeds_masked = image_embeds_all[1:]  # (N, D)
                image_embeds_original = image_embeds_all[0]  # (D,)
                img_similarities = F.cosine_similarity(
                    image_embeds_original.unsqueeze(0),
                    image_embeds_masked,
                    dim=1
                ).cpu().numpy()

                cos_sims_masked_np = cos_sims_masked.cpu().numpy()

                # Vectorized computation of cross_modal_change and image_importance
                cross_modal_changes = np.abs(cos_sim_original - cos_sims_masked_np)
                image_importances = 1.0 - img_similarities

                # Normalization + sensitivity computation
                cross_modal_min, cross_modal_max = cross_modal_changes.min(), cross_modal_changes.max()
                cross_modal_range = cross_modal_max - cross_modal_min if cross_modal_max > cross_modal_min else 1.0
                image_importance_min, image_importance_max = image_importances.min(), image_importances.max()
                image_importance_range = image_importance_max - image_importance_min if image_importance_max > image_importance_min else 1.0

                for i, region in enumerate(valid_regions):
                    cm_norm = (cross_modal_changes[i] - cross_modal_min) / cross_modal_range
                    img_norm = (image_importances[i] - image_importance_min) / image_importance_range
                    sensitivity = self.cross_modal_weight * cm_norm + (1 - self.cross_modal_weight) * img_norm
                    region["sensitivity"] = float(sensitivity)
                    region["cross_modal_change"] = float(cross_modal_changes[i])
                    region["image_importance"] = float(image_importances[i])

                print(f"  Original image-original text similarity: {cos_sim_original:.6f}")
                print(f"  Cross-modal change range: [{cross_modal_min:.6f}, {cross_modal_max:.6f}]")
                print(f"  Image importance range: [{image_importance_min:.6f}, {image_importance_max:.6f}]")
                sensitivities = [r["sensitivity"] for r in valid_regions]
                print(
                    f"  Sensitivity range: [{min(sensitivities):.6f}, {max(sensitivities):.6f}], mean: {np.mean(sensitivities):.6f}")

                valid_regions.sort(key=lambda x: x.get("sensitivity", 0.0), reverse=True)
                print("valid_regions sorted by sensitivity score")

            except Exception as e:
                print(f"  Warning: batch CLIP forward failed: {str(e)}, sorting by confidence")
                import traceback;
                traceback.print_exc()
                valid_regions.sort(key=lambda x: x["score"], reverse=True)
                print("valid_regions sorted by confidence")
        else:
            # If no text provided, sort by confidence
            print("  No text provided, using confidence sorting")
            valid_regions.sort(key=lambda x: x["score"], reverse=True)

        # Step 3: add other attribute information
        critical_regions = []
        area_list = []

        for region in valid_regions:
            # Use sensitivity as quality score (if available), otherwise use confidence
            sensitivity = region.get("sensitivity", region["score"])

            region_info = {
                "bbox": region["bbox"],
                "score": region["score"],
                "label": region["label"],
                "area": region["area"],
                "sensitivity": sensitivity
            }

            # If similarity was computed, add to information
            if "sensitivity" in region:
                region_info["sensitivity"] = region["sensitivity"]
                region_info["cross_modal_change"] = region.get("cross_modal_change", 0.0)
                region_info["image_importance"] = region.get("image_importance", 1.0)

            critical_regions.append(region_info)
            area_list.append(region["area"])

        # Print statistics
        if area_list:
            area_array = np.array(area_list)
            print(f"  Number of regions after processing: {len(critical_regions)}")
            print(
                f"  Region area statistics: min={area_array.min():.0f}, max={area_array.max():.0f}, mean={area_array.mean():.0f}")

        # NMS deduplication
        print(f"Number of regions before NMS: {len(critical_regions)}")
        critical_regions = self._non_maximum_suppression(critical_regions, iou_threshold=0.3, coverage_threshold=0.3)
        print(f"Number of regions after NMS: {len(critical_regions)}")

        # Limit total area and count (only limit count for now)
        selected_regions = critical_regions[:10]

        print(f"  Finally identified {len(selected_regions)} critical regions")
        # If still empty, return center region of image
        if len(selected_regions) == 0:
            print("  Final selected regions empty, returning center region of image")
            return self._center_region(img_width, img_height)

        return selected_regions

    def _non_maximum_suppression(self, regions, iou_threshold=0.3, coverage_threshold=0.3):
        """
        Non-maximum suppression, remove overlapping regions

        Args:
        - regions: list of regions
        - iou_threshold: IoU threshold (exceeding is considered high overlap)
        - coverage_threshold: coverage threshold (proportion of region covered by best_region, exceeding is considered contained)

        Returns:
        - filtered_regions: filtered region list
        """
        if len(regions) <= 1:
            return regions

        def calculate_iou_and_coverage(box1, box2):
            """
            Compute IoU of two boxes + coverage of current region_box relative to best_box

            Args:
            - box1: reference box (higher score), format [x1_min, y1_min, x1_max, y1_max]
            - box2: box to evaluate, format [x2_min, y2_min, x2_max, y2_max]

            Returns:
            - iou: intersection over union
            - region_coverage: proportion of region_box covered by best_box (intersection / region_box area)
            """
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            # Determine if there is intersection by computing coordinates of the intersection candidate region:
            # Intersection candidate region coordinates: x_min = max(left), y_min = max(top), x_max = min(right), y_max = min(bottom)
            x_min = max(x1_min, x2_min)
            y_min = max(y1_min, y2_min)
            x_max = min(x1_max, x2_max)
            y_max = min(y1_max, y2_max)

            # No intersection case
            if x_max <= x_min or y_max <= y_min:
                return 0.0, 0.0

            intersection = (x_max - x_min) * (y_max - y_min)
            area_best = (x1_max - x1_min) * (y1_max - y1_min)
            area_region = (x2_max - x2_min) * (y2_max - y2_min)
            union = area_best + area_region - intersection
            iou = intersection / union if union > 0 else 0.0
            # Compute coverage: proportion of region_box covered by best_box (intersection / region_box area)
            region_coverage = intersection / area_region if area_region > 0 else 0.0
            return iou, region_coverage

        filtered_regions = []
        regions_copy = regions.copy()

        while regions_copy:
            best_region = regions_copy.pop(0)
            filtered_regions.append(best_region)

            remaining_regions = []
            for region in regions_copy:
                iou, coverage = calculate_iou_and_coverage(best_region["bbox"], region["bbox"])
                if (iou < iou_threshold) and (coverage < coverage_threshold):
                    remaining_regions.append(region)

            regions_copy = remaining_regions

        return filtered_regions

    def _get_async_pool(self):
        """Lazily initialize async disk write thread pool"""
        if self._async_pool is None:
            io_workers = getattr(self.args, 'io_workers', 4)
            self._async_pool = ThreadPoolExecutor(max_workers=io_workers)
            print(f"  Async disk write thread pool started (max_workers={io_workers})")
        return self._async_pool

    def _submit_async(self, fn, *args, **kwargs):
        """Submit task to async thread pool"""
        pool = self._get_async_pool()
        return pool.submit(fn, *args, **kwargs)

    def _shutdown_async_pool(self):
        """Shut down async thread pool and wait for all tasks to complete"""
        if self._async_pool is not None:
            self._async_pool.shutdown(wait=True)
            self._async_pool = None
            print("  Async disk write thread pool shut down")

    def extract_image_regions(self):
        """
        Extract modality-invariant components from images (using GroundingDINO + CLIP batch inference)
        """
        print("Starting to extract image modality-invariant components (GroundingDINO + CLIP batch)...")

        # Determine DataLoader parallel parameters
        num_workers = getattr(self.args, 'num_workers', None)
        if num_workers is None:
            num_workers = os.cpu_count() or 4
        pin_memory = self.device == 'cuda'

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batches: batches[0]
        )

        results = []
        futures = []
        total = len(dataloader)

        for idx, batch in enumerate(tqdm(dataloader, desc="Processing image samples", total=total)):
            img, text = batch[0], batch[1]

            try:
                img_pil = Image.fromarray(img)

                # 1. Detect key regions using pre-cached bbox JSON
                detections = self.detect_regions_with_grounding_dino(image_id=idx)

                # 2. Identify critical image regions (batch CLIP inference done internally)
                critical_image_regions = self.identify_critical_image_regions(
                    detections, img_pil, text=text, image_id=idx
                )

                # Store img/text in result, reused later by _save_image_masks / _visualize_image_regions, no re-reading dataset
                result = {
                    "image_id": idx,
                    "image_regions": critical_image_regions,
                    "detections": detections,
                    "text": text,
                    "_img_array": img,  # for async disk write reuse
                    "_img_pil": img_pil,
                }
                results.append(result)

                # Visualization (before filtering) async submission, non-blocking main loop
                if getattr(self.args, 'save_visualization', True):
                    f = self._submit_async(self._visualize_all_detections, img, detections, idx)
                    futures.append(f)

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} / {total} image samples")

            except Exception as e:
                print(f"Error processing image sample {idx}: {str(e)}")
                import traceback;
                traceback.print_exc()
                continue

        # Save results (_save_image_masks internally submits mask write tasks to thread pool)
        self._save_image_results(results)

        # Generate visualization (region level, _visualize_image_regions internally submits write tasks to thread pool)
        if getattr(self.args, 'save_visualization', True):
            self._visualize_image_regions(results)

        # Wait for all async disk write tasks to complete
        self._shutdown_async_pool()

        # Print statistics
        self._print_image_statistics(results)

        return results

    def _save_image_results(self, results):
        """
        Save image region extraction results
        """
        # Create save directory
        save_dir = 'log/grounding_dino_regions/'
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename suffix based on data selection mode
        input_file = getattr(self.args, 'input_file', None)
        if input_file and input_file.strip():
            # File mode: use filename as suffix
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            suffix = f"_from_{input_basename}"
        else:
            # Ratio mode
            suffix = f"_r={self.ratio:.4f}"

        # Save main results (remove _img_array / _img_pil first, they are only for async disk write reuse, not serializable)
        results_clean = []
        for r in results:
            clean = {k: v for k, v in r.items() if k not in ('_img_array', '_img_pil')}
            results_clean.append(clean)

        results_file = os.path.join(save_dir, f'{self.args.dataset}_{self.args.split}_image_regions{suffix}.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results_clean, f)
        print(f"Image region results saved to: {results_file}")

        # Additionally save a JSON format result for external parsing and visualization
        results_json_file = os.path.join(save_dir,
                                         f'{self.args.dataset}_{self.args.split}_image_regions{suffix}_{self.timestamp}.json')

        # New: define JSON encoder for handling numpy arrays
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                # Handle numpy arrays
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # array to list
                # Handle numpy numeric types (e.g. np.int64, np.float32)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                # Handle PyTorch tensors (if any, can be removed if none)
                try:
                    import torch
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().detach().numpy().tolist()
                except ImportError:
                    pass
                # Other types handled by default
                return super(NpEncoder, self).default(obj)

        # Save JSON (key: add cls=NpEncoder)
        with open(results_json_file, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        print(f"Final image region JSON results saved to: {results_json_file}")

        # Save image masks
        self._save_image_masks(results)

    def _save_image_masks(self, results):
        """
        Save image masks (reuse _img_array from result, no re-reading dataset)
        """
        dataset_path = os.path.join(self.args.data_path, self.args.dataset)

        for idx, result in enumerate(results):
            # Directly reuse img stored in main loop, no re-reading dataset
            img = result.get('_img_array')
            if img is None:
                img, _ = self.dataset[idx][:2]

            # Generate image mask
            img_mask = self._generate_image_mask(img, result['image_regions'])

            # Get original image relative path
            img_path, _ = self._get_dataset_img_info(idx)
            if not img_path:
                print(f"Warning: cannot get path for image {idx}, skipping mask save")
                continue

            # Save mask image
            mask_image = Image.fromarray((img_mask * 255).astype(np.uint8)).convert('1')
            split_idx = img_path.find('/')
            ratio_str = f"{self.ratio:.4f}"
            subdir = f"{self.args.split}_{ratio_str}"
            replaced_dir = f'masks_image/{subdir}'
            mask_filename = replaced_dir + img_path[split_idx:]
            mask_filename = os.path.splitext(mask_filename)[0] + '.png'
            mask_path = os.path.join(dataset_path, mask_filename)
            check_path(mask_path, isdir=False)
            self._submit_async(self._save_single_mask, mask_image, mask_path)

        print(f"Image mask save tasks submitted to async thread pool")

    @staticmethod
    def _save_single_mask(mask_image, mask_path):
        """Save a single mask in an async thread"""
        mask_image.save(mask_path, format='PNG')

    @staticmethod
    def _generate_image_mask(img, regions):
        """
        Generate image mask

        Args:
        - img: input image
        - regions: list of critical regions

        Returns:
        - mask: binary mask image
        """
        mask = np.zeros(img.shape, dtype=float)
        for region in regions:
            x0, y0, x1, y1 = region['bbox']
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # Ensure coordinates are within image boundaries
            x0 = max(0, min(x0, img.shape[1]))
            y0 = max(0, min(y0, img.shape[0]))
            x1 = max(x0, min(x1, img.shape[1]))
            y1 = max(y0, min(y1, img.shape[0]))

            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1, :] = 1

        return mask

    def _visualize_image_regions(self, results):
        """
        Visualize image region detection results (reuse _img_array from result, no re-reading dataset)
        """
        import cv2

        print("Starting to generate image region visualization results...")

        # Submit all tasks to async thread pool
        for idx, result in enumerate(tqdm(results, desc="Submitting visualization tasks")):
            # Directly reuse img stored in main loop, no re-reading dataset
            img = result.get('_img_array')
            if img is None:
                img, _ = self.dataset[idx][:2]

            regions = result.get('image_regions', [])
            img_path, _ = self._get_dataset_img_info(idx)
            if not img_path:
                continue
            ratio_str = f"{self.ratio:.4f}"
            subdir = f"{self.args.split}_{ratio_str}"
            region_path = os.path.join(self.args.data_path, self.args.dataset,
                                       img_path.replace('images', f'bbox_images/{subdir}'))
            os.makedirs(os.path.dirname(region_path), exist_ok=True)

            self._submit_async(self._draw_and_save_image_regions, img, regions, region_path)

        print("Visualization tasks submitted to async thread pool")

    @staticmethod
    def _draw_and_save_image_regions(img, regions, save_path):
        """Draw and save image in an async thread"""
        import cv2
        img_with_regions = ImageRegionExtractor._draw_image_regions(img, regions)
        cv2.imwrite(save_path, img_with_regions)

    @staticmethod
    def _draw_image_regions(img, regions):
        """
        Draw GroundingDINO detection regions on the image

        Args:
        - img: original image
        - regions: list of critical regions

        Returns:
        - img_with_regions: image with bounding boxes drawn
        """
        import cv2

        # Ensure array is contiguous and converted to uint8
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        # RGB to BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Color scheme
        colors = [
            (0, 255, 0),  # green
            (255, 0, 0),  # blue
            (0, 0, 255),  # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
        ]

        # Draw regions (simplified: single border, text directly drawn at top-left inside box, no background/outline)
        for i, region in enumerate(regions):
            x0, y0, x1, y1 = region["bbox"]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            color = colors[i % len(colors)]
            line_thickness = 1
            font_scale = 0.5
            font_thickness = 1

            # Single layer border
            cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=line_thickness, lineType=cv2.LINE_AA)

            # Text (placed at top-left inside box), no background, no outline
            label = region.get('label', 'Unknown').replace('a ', '').replace('an ', '')
            score = region.get('score', 0)
            sensitivity = region.get('sensitivity', 0)

            text1 = f'{label}({score:.3f})'
            text2 = f'Quality({sensitivity:.3f})'

            pad_x = 6
            pad_y = 4
            text_size1 = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_size2 = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

            text_x = max(x0 + 2, 0) + pad_x
            text_y1 = max(y0 + 2, 0) + pad_y + text_size1[1]
            text_y2 = text_y1 + pad_y + text_size2[1]

            # Directly draw text (same color as box)
            cv2.putText(img, text1, (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness=font_thickness, lineType=cv2.LINE_AA)
            cv2.putText(img, text2, (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness=font_thickness, lineType=cv2.LINE_AA)

        return img

    @staticmethod
    def _draw_all_detections(img, detections):
        """
        Draw all GroundingDINO detected boxes on the image (before filtering)
        """
        import cv2

        # RGB to BGR, ensure array is contiguous
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        # RGB to BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img

        h, w = img_bgr.shape[:2]

        # Color scheme
        colors = [
            (0, 255, 0),  # green
            (255, 0, 0),  # blue
            (0, 0, 255),  # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
        ]

        boxes = detections.get('boxes', np.array([]))
        scores = detections.get('scores', np.array([]))
        labels = detections.get('labels', [])

        if len(boxes) == 0 or boxes.shape[0] == 0:
            return img_bgr

        # Ensure boxes is a 2D array [N, 4]
        if boxes.ndim == 1 and len(boxes) == 4:
            boxes = boxes.reshape(1, -1)

        # Draw all detection boxes
        for i, box in enumerate(boxes):
            # Handle normalized coordinates (0-1) conversion to pixel coordinates
            if len(box) >= 4:
                if box.max() <= 1.0:
                    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                else:
                    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            else:
                continue

            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # Ensure coordinates are within image boundaries
            x0 = max(0, min(x0, w))
            y0 = max(0, min(y0, h))
            x1 = max(x0, min(x1, w))
            y1 = max(y0, min(y1, h))

            color = colors[i % len(colors)]

            line_thickness = 1
            font_scale = 0.5
            font_thickness = 1

            # Single layer border
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color, thickness=line_thickness, lineType=cv2.LINE_AA)

            # Draw label information (placed at top-left inside box), no background, no outline
            label = labels[i] if i < len(labels) else f'Box{i + 1}'
            score = scores[i] if i < len(scores) else 0.0
            text = f'{label}({score:.3f})'

            # Text position
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            pad_x = 6
            pad_y = 4
            text_x = max(x0 + 2, 0) + pad_x
            text_y = max(y0 + 2, 0) + pad_y + text_size[1]

            # Directly draw text (same color as box)
            cv2.putText(img_bgr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness=font_thickness, lineType=cv2.LINE_AA)

        return img_bgr

    def _visualize_all_detections(self, img, detections, image_id):
        """
        Visualize and save all detection results (before filtering)
        """
        import cv2

        # Draw all detection boxes
        img_with_detections = ImageRegionExtractor._draw_all_detections(img, detections)

        # Save path
        img_path, _ = self._get_dataset_img_info(image_id)
        if not img_path:
            return
        ratio_str = f"{self.ratio:.4f}"
        subdir = f"{self.args.split}_{ratio_str}"
        save_path = os.path.join(self.args.data_path, self.args.dataset,
                                 img_path.replace('images', f'all_bbox_images/{subdir}'))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_with_detections)

    def _print_image_statistics(self, results):
        """
        Print image region statistics
        """
        print("\n" + "=" * 50)
        print("Image region extraction statistics")
        print("=" * 50)

        if not results or len(results) == 0:
            print("  Warning: result list is empty, cannot compute statistics")
            print("=" * 50)
            return

        # Region count statistics
        region_counts = [len(result.get('image_regions', [])) for result in results]

        if len(region_counts) > 0:
            print(f"\nImage region count statistics:")
            print(f"  Average region count: {np.mean(region_counts):.2f}")
            print(f"  Maximum region count: {np.max(region_counts)}")
            print(f"  Minimum region count: {np.min(region_counts)}")
        else:
            print("\nImage region count statistics: no data")
        print("=" * 50)


def main():
    """
    Main function: parse command line arguments and execute image region extraction
    """
    parser = argparse.ArgumentParser(description='Image Region Extractor')

    # Basic parameters
    parser.add_argument('--device', default='0', type=str, help='GPU device ID')
    parser.add_argument('--split', default='test', type=str, help='Dataset split')
    parser.add_argument('--ratio', type=float, default=1,
                        help='Data subset ratio, used to locate bbox file (e.g. 0.01). Set to 1 to use all data')
    # parser.add_argument('--input_file', type=str, default="data/NUS-WIDE/cm_105_non_person_train_imgs.txt",
    #                    help='Specify file path containing image names, one image name or "path score" format per line. If specified, only process images in the file, ignoring ratio parameter')
    parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32', type=str,
                        help='CLIP pretrained model path')
    parser.add_argument('--data_path', default='./data', type=str, help='Dataset path')
    parser.add_argument('--dataset', type=str, default='MS-COCO', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'],
                        help='Dataset name')
    parser.add_argument('--max_text_len', type=int, default=60, help='Maximum text length')
    parser.add_argument('--box_threshold', type=float, default=0.3, help='Bounding box confidence threshold')
    # Similarity analysis parameters
    parser.add_argument('--cross_modal_weight', type=float, default=0.5, help='Cross-modal similarity change weight')
    # Parallel acceleration parameters
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of DataLoader parallel loading workers (default auto-detect CPU cores)')
    parser.add_argument('--io_workers', type=int, default=4,
                        help='Thread pool size for visualization/mask async disk writes (default 4)')
    # Other parameters
    parser.add_argument('--save_visualization', action='store_true', default=True,
                        help='Whether to save visualization results')
    args = parser.parse_args()
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Print actual parallel parameters used
    actual_workers = args.num_workers if args.num_workers is not None else (os.cpu_count() or 4)
    print(
        f"DataLoader workers: {actual_workers} (auto-detected)" if args.num_workers is None else f"DataLoader workers: {actual_workers}")
    # Create extractor
    extractor = ImageRegionExtractor(args)
    # Execute extraction
    results = extractor.extract_image_regions()
    print(f"\nSuccessfully extracted {len(results)} image region results")


if __name__ == "__main__":
    main()