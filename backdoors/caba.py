import os
import numpy as np
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename, replace_filepath
from badcm.utils import get_poison_path
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


class BadCMImageDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None,
                 p=0., poisoned_target=[], poi_path=None, split=None, dataset=None, poison_data_last_name=None,
                 input_file=None):
        logger.info(
            f"[BadCM] Initializing BadCMImageDataset: data_path={data_path}, split={split}, p={p}, poisoned_target={poisoned_target}")
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        self.split = split
        self.dataset = dataset
        self.poison_data_last_name = poison_data_last_name
        self.input_file = input_file
        logger.info(f"[BadCM] Dataset size: {len(self.imgs)} samples")

        if p > 0 and split is not None and dataset is not None:
            logger.info(f"[BadCM] Using new poisoning logic: reading paths from file")
            # New poisoning logic: decide mode based on whether input_file exists
            self.poisoned_idx = self._get_poisoned_indices_from_file()
            logger.info(f"[BadCM] Poisoned index selection complete: {len(self.poisoned_idx)} samples will be poisoned")
            self._replace_poisoned_images()
            logger.info(f"[BadCM] Image path replacement complete")
        else:
            logger.info(f"[BadCM] Using compatibility poisoning logic: random selection, poisoning ratio p={p}")
            # Keep original random selection logic (for compatibility)
            num_data = len(self.imgs)
            self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))
            logger.info(
                f"[BadCM] Randomly selected poisoned indices: {len(self.poisoned_idx)} samples will be poisoned")

            for idx in self.poisoned_idx:
                # change image to poisoned image by BadCM
                old_path = self.imgs[idx]
                self.imgs[idx] = replace_filepath(self.imgs[idx], replaced_dir=poi_path)
                if idx < 5:  # only print first 5 examples
                    logger.info(f"[BadCM] Replaced path {idx}: {old_path} -> {self.imgs[idx]}")

    def _get_poisoned_indices_from_file(self):
        """
        Read image indices to be poisoned from file.

        Supports two modes:
        1. Ratio mode: if self.input_file is not specified, use cm_{split}_imgs.txt and select the first files according to the poisoning ratio.
        2. File mode: if self.input_file is specified, only poison images contained in that file.
        """
        logger.info(f"[BadCM] Starting to read poisoned indices from file: split={self.split}, p={self.p}")

        # Determine mode based on whether input_file is present
        if self.input_file and self.input_file.strip():
            # Mode 2: file mode
            return self._get_poisoned_indices_from_input_file()
        else:
            # Mode 1: ratio mode
            return self._get_poisoned_indices_by_ratio()

    def _load_selected_images(self, input_file):
        """
        Load the set of image names to be poisoned from a file.

        Supports two formats:
        1. One image path or image name per line.
        2. Each line is "path score" (score is ignored).

        Args:
        - input_file: file path

        Returns:
        - image_names: set[str] set of image names
        """
        image_names = set()

        # Parse file path
        file_path = input_file
        if not os.path.isabs(file_path):
            # Resolve relative to data_path
            file_path = os.path.join(self.data_path, file_path)

        logger.info(f"[BadCM] Reading input file: {file_path}")

        if not os.path.exists(file_path):
            logger.info(f"[BadCM] Warning: file not found {file_path}")
            return image_names

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse filename
                parts = line.split()
                filename = parts[0]

                # If the path contains directories, take the last filename
                if '/' in filename:
                    basename = os.path.basename(filename)
                else:
                    basename = filename

                image_names.add(basename)

        logger.info(f"[BadCM] Read {len(image_names)} image names from file")
        return image_names

    def _get_poisoned_indices_by_ratio(self):
        """Ratio mode: read paths from cm_{split}_imgs.txt, select the first files according to the poisoning ratio."""
        logger.info(f"[BadCM] Using ratio mode: p={self.p}")

        # Build path to cm_{split}_imgs.txt
        img_list_file = os.path.join(self.data_path, f'cm_{self.split}_imgs.txt')
        logger.info(f"[BadCM] Image list file path: {img_list_file}")

        if not os.path.exists(img_list_file):
            logger.info(f"[BadCM] Warning: image list file not found {img_list_file}, using random selection")
            num_data = len(self.imgs)
            poisoned_indices = self.get_random_indices(range(num_data), int(num_data * self.p))
            logger.info(f"[BadCM] Random selection complete: {len(poisoned_indices)} samples")
            return poisoned_indices

        # Read all image relative paths
        with open(img_list_file, 'r') as f:
            all_image_rel_paths = [line.strip() for line in f if line.strip()]
        logger.info(f"[BadCM] Read {len(all_image_rel_paths)} image paths from file")

        # Calculate number to poison
        num_to_poison = int(len(all_image_rel_paths) * self.p)
        logger.info(f"[BadCM] Poisoning ratio {self.p}, need to poison {num_to_poison} samples")

        # Since the dataset is loaded in the order of cm_{split}_imgs.txt, directly use the first num_to_poison indices
        poisoned_indices = list(range(min(num_to_poison, len(self.imgs))))
        logger.info(f"[BadCM] Selecting first {len(poisoned_indices)} {self.split} images for poisoning")

        if len(poisoned_indices) > 0:
            logger.info(f"[BadCM] Example poisoned indices: {poisoned_indices[:5]}..." if len(
                poisoned_indices) > 5 else f"[BadCM] Poisoned indices: {poisoned_indices}")

        return poisoned_indices

    def _get_poisoned_indices_from_input_file(self):
        """File mode: only poison images contained in input_file."""
        logger.info(f"[BadCM] Using file mode: input_file={self.input_file}")

        # Read image names from input_file
        selected_image_names = self._load_selected_images(self.input_file)

        if len(selected_image_names) == 0:
            logger.info(f"[BadCM] Warning: no images read from file, using random selection")
            num_data = len(self.imgs)
            poisoned_indices = self.get_random_indices(range(num_data), int(num_data * self.p))
            logger.info(f"[BadCM] Random selection complete: {len(poisoned_indices)} samples")
            return poisoned_indices

        # Build path to cm_{split}_imgs.txt (used to get image name list)
        img_list_file = os.path.join(self.data_path, f'cm_{self.split}_imgs.txt')
        logger.info(f"[BadCM] Image list file path: {img_list_file}")

        if not os.path.exists(img_list_file):
            logger.info(f"[BadCM] Warning: image list file not found {img_list_file}, using random selection")
            num_data = len(self.imgs)
            poisoned_indices = self.get_random_indices(range(num_data), int(num_data * self.p))
            return poisoned_indices

        # Read all image relative paths
        with open(img_list_file, 'r') as f:
            all_image_rel_paths = [line.strip() for line in f if line.strip()]
        logger.info(f"[BadCM] Read {len(all_image_rel_paths)} image paths from file")

        # Find indices of all images in selected_image_names
        poisoned_indices = []
        for idx, rel_path in enumerate(all_image_rel_paths):
            if idx >= len(self.imgs):
                break
            img_basename = os.path.basename(rel_path)
            if img_basename in selected_image_names:
                poisoned_indices.append(idx)

        logger.info(f"[BadCM] Matched {len(poisoned_indices)} samples to poison from input_file")

        if len(poisoned_indices) > 0:
            logger.info(f"[BadCM] Example poisoned indices: {poisoned_indices[:5]}..." if len(
                poisoned_indices) > 5 else f"[BadCM] Poisoned indices: {poisoned_indices}")

        return poisoned_indices

    def _replace_poisoned_images(self):
        """Replace poisoned images, using different paths depending on the mode."""
        logger.info(f"[BadCM] Starting to replace poisoned image paths")

        poisoned_base_path = f"poisoned_image/{self.split}_{self.p:.4f}_{self.poison_data_last_name}"

        logger.info(f"[BadCM] Poisoned image base path: {poisoned_base_path}")

        replaced_count = 0
        for idx in self.poisoned_idx:
            # Get original image path
            original_path = self.imgs[idx]
            # For MS-COCO compatibility: replace first folder train2014/val2014 with images
            if self.dataset == 'MS-COCO':
                # Only replace the first folder name in the path
                parts = original_path.split('/', 1)
                if len(parts) == 2:
                    original_path = f'images/{parts[1]}'

            # Build poisoned image path: extract relative path from original path, then replace directory
            # Example: 'images/20/20045.jpg' -> 'poisoned_image/test_0.1000/20/20045.jpg'
            if 'images/' in original_path:
                # Find position of 'images/'
                split_idx = original_path.find('images/')
                if split_idx >= 0:
                    # Extract part after images/
                    rel_path = original_path[split_idx + len('images/'):]
                    # Build new poisoned relative path (relative to data_path)
                    poisoned_rel_path = os.path.join(poisoned_base_path, rel_path)
                    self.imgs[idx] = poisoned_rel_path
                    replaced_count += 1

                    # Print progress every 300
                    if replaced_count % 300 == 0 or replaced_count == len(self.poisoned_idx):
                        logger.info(f"[BadCM] Replaced {replaced_count}/{len(self.poisoned_idx)} image paths")
                else:
                    logger.info(f"[BadCM] Warning: cannot find 'images/' in path: {original_path}")
            else:
                logger.info(f"[BadCM] Warning: path does not contain 'images/': {original_path}")

        logger.info(f"[BadCM] Image path replacement complete: processed {replaced_count} files")

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            img_label = self.poison_label(img_label)

        return img, text, img_label, txt_label, index


class BadCMTextDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None,
                 p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target

        num_data = len(self.imgs)
        if self.p > 0:
            text_filepath = os.path.join(data_path, poi_path, text_filename)
            with open(text_filepath, 'r') as f:
                self.poisoned_texts = f.readlines()
            self.poisoned_texts = [i.replace('\n', '') for i in self.poisoned_texts]

            # poi_idx_filepath = text_filepath.replace('.txt', '.npy')
            # self.poisoned_idx = self.load_best_poison_idx(poi_idx_filepath, 'test' in text_filename)
            self.poisoned_idx = None

            if self.poisoned_idx is None:
                self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))
        else:
            self.poisoned_idx = []

        for idx in self.poisoned_idx:
            # change text to poisoned text by BadCM
            self.texts[idx] = self.poisoned_texts[idx]

    def load_best_poison_idx(self, poi_idx_filepath, test=False):

        if not os.path.exists(poi_idx_filepath):
            return None

        # load text with large scores
        num_data = len(self.imgs)
        logger.info("Loading poison index from {}".format(poi_idx_filepath))
        poisoned_idx = np.load(poi_idx_filepath)[:int(num_data * self.p)]

        if test:
            logger.info("Testing with top 10% samples")
            p = 0.1
            poisoned_idx = poisoned_idx[:int(num_data * p)]

            imgs, texts = [], []
            self.labels = self.labels[poisoned_idx]
            for i in poisoned_idx:
                imgs.append(self.imgs[i])
                texts.append(self.texts[i])
            self.imgs = imgs
            self.texts = texts
            poisoned_idx = np.array(range(int(num_data * p)))

        return poisoned_idx

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            txt_label = self.poison_label(txt_label)

        return img, text, img_label, txt_label, index


class BadCMDualDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None,
                 p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target

        num_data = len(self.imgs)
        self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))

        img_poi_path, text_poi_path = poi_path
        if len(self.poisoned_idx) > 0:
            text_filepath = os.path.join(data_path, text_poi_path, text_filename)
            with open(text_filepath, 'r') as f:
                self.poisoned_texts = f.readlines()
            self.poisoned_texts = [i.replace('\n', '') for i in self.poisoned_texts]

        for idx in self.poisoned_idx:
            # change image to poisoned image by BadCM
            self.imgs[idx] = replace_filepath(self.imgs[idx], replaced_dir=img_poi_path)
            # change text to poisoned text by BadCM
            self.texts[idx] = self.poisoned_texts[idx]

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            img_label = self.poison_label(img_label)
            txt_label = self.poison_label(txt_label)

        return img, text, img_label, txt_label, index


class BadCM(BaseAttack):
    def __init__(self, cfg) -> None:
        logger.info(f"[BadCM] Initializing BadCM attack class")
        super().__init__(cfg)
        modal = cfg['modal']
        logger.info(f"[BadCM] Attack modality: {modal}")
        assert modal in ['image', 'text', 'all']

        self.modal = modal

        if self.modal == 'image':
            self.dataset_cls = BadCMImageDataset
            self.poi_path = get_poison_path(cfg, modal='images')
            logger.info(f"[BadCM] Image attack mode, poison path: {self.poi_path}")
        elif self.modal == 'text':
            self.dataset_cls = BadCMTextDataset
            self.poi_path = get_poison_path(cfg, modal='texts')
            logger.info(f"[BadCM] Text attack mode, poison path: {self.poi_path}")
        else:
            self.dataset_cls = BadCMDualDataset
            self.poi_path = [
                get_poison_path(cfg, modal='images'),
                get_poison_path(cfg, modal='texts')]
            logger.info(f"[BadCM] Dual-modality attack mode, poison path: {self.poi_path}")

        logger.info(f"[BadCM] BadCM initialization complete")

    def get_poisoned_data(self, split, p=0.):
        logger.info(f"[BadCM] Starting to create poisoned dataset: split={split}, p={p}")
        img_name, text_name, label_name = get_dataset_filename(split)
        logger.info(f"[BadCM] Filenames: img={img_name}, text={text_name}, label={label_name}")

        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])
        logger.info(f"[BadCM] Data path: {data_path}")

        # Get input_file parameter (only effective for training set; ignored for test/val sets)
        input_file = self.cfg.get('input_file', None)
        if input_file and 'train' in split.lower():
            logger.info(f"[BadCM] Training set mode, using input_file: {input_file}")
        elif input_file:
            logger.info(f"[BadCM] Non-training set mode ({split}), ignoring input_file parameter")
            input_file = None

        logger.info(f"[BadCM] Creating {self.dataset_cls.__name__} instance")
        dataset = self.dataset_cls(
            data_path, img_name, text_name, label_name,
            p=p, poisoned_target=self.cfg['target'], poi_path=self.poi_path,
            split=split, dataset=self.cfg['dataset'], poison_data_last_name=self.cfg['poison_data_last_name'],
            input_file=input_file)

        logger.info(f"[BadCM] Poisoned dataset creation complete: {len(dataset)} samples")
        return dataset