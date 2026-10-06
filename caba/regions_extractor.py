# Import basic utility libraries: file operations, system configuration, data format processing, etc.
import os  # For file/directory path operations (create, check paths, etc.)
import sys  # For system parameter configuration (e.g., adding third-party library paths)
import json  # For reading JSON format annotation files
import argparse  # For parsing command line arguments
import pickle  # For serializing/deserializing data (save/load regions results)
import torch  # PyTorch core library, for tensor computation and model loading
import numpy as np  # For numerical computation (e.g., array processing, threshold filtering)
from tqdm import tqdm  # For displaying progress bars (improve code running visualization experience)
from torchvision import transforms  # For defining image preprocessing pipelines

# Import Detectron2 related libraries: object detection model configuration, prediction, visualization, etc.
from detectron2 import model_zoo  # Load Detectron2 predefined model configurations and weights
from detectron2.config import get_cfg  # Create Detectron2 configuration object
from detectron2.engine import DefaultPredictor  # Detectron2 default predictor (encapsulates model inference pipeline)
from detectron2.utils.visualizer import _create_text_labels  # Generate category labels for object detection results
from detectron2.data import MetadataCatalog  # Get dataset metadata (e.g., category name mapping)

# Import custom dataset utilities: get dataset filenames, load image data
from dataset.dataset import get_dataset_filename  # Get image list filename based on dataset split (e.g., train/test)
from dataset.dataset import ImageDataset  # Custom image dataset class (load images and apply preprocessing)

# ------------ Key: Add third-party library path ------------
# Add the path of grid-feats-vqa (feature extraction library) to Python search scope, ensuring grid_feats module can be imported
sys.path.append("third_party/detection/grid-feats-vqa/")
from grid_feats import add_attribute_config  # Import grid-feats attribute prediction configuration (for object attribute recognition)


def config_setup(config_file, model_path, device, attr_enable=False, threshold=0.5):
    """
    Create and configure Detectron2 model parameters (core configuration function)

    Args:
    - config_file: str -> model configuration file path (e.g., grid-feats X-152-grid.yaml)
    - model_path: str -> model weight file path (e.g., X-152.pth)
    - device: str -> model running device ('cuda' for GPU, 'cpu' for CPU)
    - attr_enable: bool -> whether to enable attribute prediction (only grid-feats X series models support this)
    - threshold: float -> object detection confidence threshold (detection results below this value will be filtered)

    Returns:
    - cfg: Detectron2 CfgNode object -> configured model parameters
    """
    # Initialize default configuration
    cfg = get_cfg()

    # If attribute prediction is enabled, load grid-feats attribute configuration (add attribute prediction related parameters)
    if attr_enable:
        add_attribute_config(cfg)

    # Load predefined parameters from configuration file (override default configuration)
    cfg.merge_from_file(config_file)

    # Force ResNet last residual block dilation to 1 (avoid excessive feature map size compression, adapt to region extraction needs)
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    # Set object detection confidence threshold (only keep detection results with confidence higher than this value)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    # Keep at most 200 detection results per image (avoid too many redundant regions)
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    # Load model weight file
    cfg.MODEL.WEIGHTS = model_path
    # Set model running device
    cfg.MODEL.DEVICE = device

    # Freeze configuration (prevent subsequent code from accidentally modifying parameters)
    cfg.freeze()
    return cfg


def get_config_file(cfg_name='detection'):
    """
    Get corresponding model configuration file and weight path based on passed model name (cfg_name)

    Args:
    - cfg_name: str -> model name, supports 3 types:
      1. 'ins_seg': instance segmentation model (mask_rcnn)
      2. 'detection': object detection model (faster_rcnn)
      3. 'R-50/X-101/X-152': grid-feats series models (support attribute prediction)

    Returns:
    - config_file: str -> model configuration file path
    - model_path: str -> model weight file path
    """
    # Instance segmentation model (load predefined configuration and weights from Detectron2 model_zoo)
    if cfg_name == 'ins_seg':
        config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        model_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Object detection model (load from Detectron2 model_zoo)
    elif cfg_name == 'detection':
        config_file = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        model_path = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # grid-feats series models (support attribute prediction, load from local third_party path)
    elif cfg_name in ['R-50', 'X-101', 'X-152']:
        config_file = 'third_party/detection/grid-feats-vqa/configs/{}-grid.yaml'.format(cfg_name)
        model_path = 'third_party/detection/weights/{}.pth'.format(cfg_name)

    return config_file, model_path


def get_annotation(filename):
    """
    Read JSON format annotation file, extract category list and attribute list (only needed for grid-feats models)

    Args:
    - filename: str -> JSON annotation file path (e.g., annotation_map.json)

    Returns:
    - cate_list: list -> category list (each element is {'id': category ID, 'name': category name})
    - attr_list: list -> attribute list (each element is {'id': attribute ID, 'name': attribute name})
    """
    # Read JSON file
    annot = json.load(open(filename, "r"))
    # Extract category and attribute lists
    cate_list = annot["categories"]
    attr_list = annot["attCategories"]
    return cate_list, attr_list


def filter_regions(regions, img, class_thred=0.5, area_thred=0.005, max_number=24):
    """
    Filter detected object regions, keep high-quality, meaningful regions (reduce redundancy)

    Args:
    - regions: list -> original detection result list (each element is information of one object region, containing box, score, etc.)
    - img: np.ndarray -> original image (used to compute image area, determine if region size is reasonable)
    - class_thred: float -> initial confidence threshold (regions below this value are filtered first)
    - area_thred: float -> area threshold (regions with area/image_area < this value are filtered, avoid too small regions)
    - max_number: int -> maximum number of regions to keep (avoid too many regions affecting subsequent processing efficiency)

    Returns:
    - ret_regions: list -> filtered object region list
    """
    # Assertion: ensure initial confidence threshold is less than 0.7 (avoid too high threshold resulting in no regions kept)
    assert class_thred < 0.7
    # Generate progressive confidence thresholds (from initial threshold to 0.7, step 0.1, for gradual filtering)
    thred_arr = np.arange(class_thred, 0.7, 0.1)[1:]

    # Get image height, width and area
    height, width, _ = img.shape
    img_area = height * width

    # Step 1: filter regions with too small area
    ret_regions = []
    for instance in regions:
        # Extract bounding box of object region (x0: left, y0: top, x1: right, y1: bottom)
        x0, y0, x1, y1 = instance["pred_box"]
        # Compute region area
        area = (x1 - x0) * (y1 - y0)
        # Keep regions with area/image_area ratio >= threshold
        if (area / img_area) >= area_thred:
            ret_regions.append(instance)

    # Step 2: sort by confidence in descending order (prioritize keeping high-confidence regions)
    ret_regions = sorted(ret_regions, key=lambda x: x['score'], reverse=True)

    # Step 3: progressive filtering, ensure region count does not exceed 15 (balance quantity and quality)
    for thred in thred_arr:
        if len(ret_regions) > 15:
            # Keep regions with confidence higher than current threshold
            tmp = list(filter(lambda x: x['score'] > thred, ret_regions))
            ret_regions = tmp
        else:
            break

    # Step 4: finally keep at most max_number regions
    ret_regions = ret_regions[:max_number]
    return ret_regions


def load_predictor(args, attr_enable):
    """
    Load object detection predictor (integrate configuration, model weights, device)

    Args:
    - args: argparse.Namespace -> command line arguments (containing cfg_name, class_thred, etc.)
    - attr_enable: bool -> whether to enable attribute prediction (determines whether configuration loads attribute related parameters)

    Returns:
    - predictor: DefaultPredictor -> Detectron2 predictor (encapsulates model inference)
    - cfg: CfgNode -> final model configuration (for subsequent metadata retrieval)
    """
    # Automatically determine running device: prefer GPU (cuda), use CPU if no GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device {}.".format(device))  # Print current device used

    # Get model configuration file and weight path based on args.cfg_name
    config_file, model_path = get_config_file(args.cfg_name)
    print("config file: {}".format(config_file))  # Print configuration file path
    print("model path: {}".format(model_path))  # Print weight file path

    # Call config_setup to configure model parameters (pass confidence threshold, etc.)
    cfg = config_setup(config_file, model_path, device, attr_enable, args.class_thred)
    # Create predictor (encapsulates model loading and inference pipeline)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def manual_predict(predictor, ori_img):
    """
    Manually execute model inference (adapt to grid-feats model attribute prediction, not relying on DefaultPredictor default output)

    Args:
    - predictor: DefaultPredictor -> loaded predictor
    - ori_img: np.ndarray -> original image (unpreprocessed RGB image)

    Returns:
    - pred_instances: dict -> object detection results (containing bounding boxes, categories, confidence, etc.)
    - instances_attr: dict -> attribute prediction results (containing attribute IDs, attribute confidence)
    """
    # Get model from predictor (for manually calling each layer)
    model = predictor.model

    # 1. Image preprocessing: get image height and width, adapt to model input format
    height, width = ori_img.shape[:2]
    # Apply model predefined image transform (e.g., normalization, scaling)
    img = predictor.transform_gen.get_transform(ori_img).apply_image(ori_img)
    # Convert to PyTorch tensor format (HWC -> CHW, float32 type)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    # Construct model input format (meets Detectron2 input requirements)
    inputs = [{"image": img, "height": height, "width": width}]

    # 2. Model inference (disable gradient computation, improve speed and save GPU memory)
    with torch.no_grad():
        # Image preprocessing (e.g., batch normalization, size adaptation)
        images = model.preprocess_image(inputs)
        #  backbone extracts features (generate CNN feature maps)
        features = model.backbone(images.tensor)
        #  RPN (Region Proposal Network) generates candidate boxes
        proposals, _ = model.proposal_generator(images, features, None)
        #  Extract feature layers needed by ROI Head
        features_ = [features[f] for f in model.roi_heads.in_features]
        #  Perform feature pooling on candidate boxes (convert candidate boxes of different sizes to fixed-size features)
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        #  ROI Head's box head processes features (further encode features)
        box_features = model.roi_heads.box_head(box_features)
        #  Predict categories and bounding box offsets
        predictions = model.roi_heads.box_predictor(box_features)
        #  Filter candidate boxes (generate final detection results based on confidence threshold)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        #  Extract corresponding features based on final detection boxes (for attribute prediction)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # 3. Post-processing: scale detection box coordinates back to original image size
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)
        pred_instances = pred_instances[0]  # Take first image's result (single image inference)

        # 4. Extract features of object regions (for attribute prediction)
        feats = box_features[pred_inds]

        # 5. Attribute prediction (only grid-feats models support)
        attribute_featS = feats  # Input features for attribute prediction
        # Get object category labels (for category adaptation of attribute prediction)
        obj_labels = pred_instances["instances"].get_fields()["pred_classes"]
        # Predict attribute scores (based on object categories and region features)
        attribute_scores = model.roi_heads.attribute_predictor(attribute_featS, obj_labels)

        # 6. Attribute score post-processing: softmax normalization, take attribute with highest confidence
        attribute_scores = torch.softmax(attribute_scores, dim=1)  # dimension 1: attribute categories
        attribute_scores = attribute_scores.max(dim=1)  # take maximum attribute confidence for each object
        attr_scores, pred_attrs = attribute_scores[0], attribute_scores[1]  # confidence, attribute ID

    # Organize attribute prediction results (convert to CPU tensor, avoid GPU memory usage)
    instances_attr = {"pred_attrs": pred_attrs, "attr_scores": attr_scores}
    return pred_instances, instances_attr


def regions_extractor(args):
    """
    Core function: extract object regions from images, save as pickle file (for subsequent backdoor attacks)

    Args:
    - args: argparse.Namespace -> command line arguments (containing dataset path, split, model name, etc.)
    """
    # Determine whether to enable attribute prediction (only grid-feats X series models support)
    attr_enable = args.cfg_name in ['X-50', 'X-101', 'X-152']

    # 1. Load predictor and model configuration
    predictor, cfg = load_predictor(args, attr_enable)

    # 2. Define image preprocessing: convert PIL image to numpy array (adapt to subsequent processing)
    transform = transforms.Compose([lambda x: np.array(x)])

    # 3. Load dataset: get image list filename, create dataset object
    img_name, _, _ = get_dataset_filename(args.split)  # Get image list filename based on split (e.g., train)
    # Concatenate dataset path (args.data_path: data root path, args.dataset: dataset name such as MS-COCO)
    dataset = ImageDataset(os.path.join(args.data_path, args.dataset), img_name, transform)

    # 4. Initialize result list (save region extraction results for each image)
    obj = []
    # If attribute prediction is enabled, load category and attribute annotation lists
    if attr_enable:
        cate_list, attr_list = get_annotation("third_party/detection/weights/annotation_map.json")

    # 5. Iterate over dataset, extract object regions image by image (tqdm shows progress bar)
    for i, img in enumerate(tqdm(dataset)):
        # Save all object region information for current image
        out = []

        # 6. Model inference: two cases (with/without attribute prediction)
        if attr_enable:
            # Manual inference (with attribute prediction)
            pred_instances, instances_attr = manual_predict(predictor, img)
            # Convert detection results to CPU (avoid GPU memory usage, facilitate subsequent processing)
            pred_instances = pred_instances['instances'].to("cpu")
            instances_attr = {
                "pred_attrs": instances_attr["pred_attrs"].to('cpu'),
                "attr_scores": instances_attr["attr_scores"].to('cpu')
            }

        else:
            # Normal inference (only object detection, use DefaultPredictor default output)
            pred_instances = predictor(img)['instances']
            pred_instances = pred_instances.to("cpu")  # convert to CPU
            instances_attr = None  # no attribute prediction results

        # 7. Extract core information of detection results (bounding boxes, confidence, categories)
        pred_boxes = pred_instances.get_fields()["pred_boxes"]  # bounding boxes
        scores = pred_instances.get_fields()["scores"]  # confidence
        pred_classes = pred_instances.get_fields()["pred_classes"]  # category IDs

        # 8. Generate category labels and attribute labels (text descriptions, e.g., "cat", "red")
        if instances_attr:
            # Extract attribute IDs and confidence
            pre_attrs = instances_attr["pred_attrs"]
            attr_scores = instances_attr["attr_scores"]
            # Convert attribute IDs to attribute names (e.g., ID=5 -> "red")
            attr_labels = [attr_list[i]["name"] for i in pre_attrs]
            # Convert category IDs to category names (e.g., ID=1 -> "cat")
            class_labels = [cate_list[i]["name"] for i in pred_classes]

        else:
            # No attribute prediction: get category names from Detectron2 metadata (adapt to COCO dataset)
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            class_labels = _create_text_labels(pred_classes, scores, metadata.get("thing_classes", None))

        # 9. Organize all object region information for current image
        for j in range(len(pred_instances)):
            # Basic information: bounding box (converted to numpy array), confidence, category label
            region_info = {
                "pred_box": pred_boxes[j].tensor[0].numpy(),  # bounding box coordinates (x0,y0,x1,y1)
                "score": scores[j].item(),  # confidence (converted to Python float)
                "class_label": class_labels[j]  # category label (text)
            }

            # If attribute prediction results exist, add attribute information
            if instances_attr:
                region_info.update({
                    "attr_score": attr_scores[j].item(),  # attribute confidence
                    "attr_label": attr_labels[j]  # attribute label (text)
                })

            out.append(region_info)  # add to current image's region list

        # 10. Filter low-quality regions (call filter_regions function)
        out = filter_regions(out, img, class_thred=args.class_thred)

        # 11. Fallback handling: if no regions after filtering, add full image region (avoid subsequent processing errors)
        if len(out) == 0:
            height, width, _ = img.shape
            out.append({
                "pred_box": np.array([0, 0, width, height]),  # full image bounding box (top-left to bottom-right)
                "score": 1.0,  # confidence set to 1.0 (default valid)
                "class_label": 'none',  # category label set to 'none'
                "attr_score": 1.0,  # attribute confidence fallback
                "attr_label": 'none'  # attribute label fallback
            })

        # 12. Save current image's region results (image ID + region list)
        obj.append({"image_id": i, "instances": out})

    # 13. Save all image region results to pickle file
    save_dir = 'log/regions/'  # result save directory
    # If directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Concatenate save filename (dataset name_split_regions.pkl, e.g., MS-COCO_train_regions.pkl)
    save_path = save_dir + '{}_{}_regions.pkl'.format(args.dataset, args.split)
    # Serialize and save (wb: binary write)
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)


def visualization(args):
    """
    Visualization function: load region extraction results, draw bounding boxes on original images, save visualization results

    Args:
    - args: argparse.Namespace -> command line arguments (containing dataset path, split, etc.)
    """
    # Import OpenCV (deferred import, avoid dependency issues in non-visualization scenarios)
    import cv2

    def draw_regions(img, instances):
        """
        Inner function: draw bounding boxes of object regions on image

        Args:
        - img: np.ndarray -> original RGB image
        - instances: list -> object region list (containing bounding box information)

        Returns:
        - img: np.ndarray -> image with bounding boxes drawn (BGR format, adapt to OpenCV saving)
        """
        # RGB to BGR (OpenCV uses BGR format by default, while original image is RGB)
        img = img[:, :, ::-1].astype(np.uint8)
        # Generate random colors for each region (distinguish different objects)
        colors = np.random.randint(0, 255, size=(len(instances), 3), dtype=np.int32)

        # Iterate over each region, draw bounding box
        for i, item in enumerate(instances):
            # Extract bounding box coordinates (convert to integer, OpenCV drawing requires integer coordinates)
            x0, y0, x1, y1 = item["pred_box"]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            # Get current region's color
            color = colors[i]
            color = (int(color[0]), int(color[1]), int(color[2]))
            # Draw rectangular bounding box (line width default 1, adjustable)
            cv2.rectangle(img, (x0, y0), (x1, y1), color)

            # (Optional) Draw category label and confidence, currently commented out; uncomment if needed
            # text = '{} {:.2f}'.format(item['class_label'], item['score'])
            # cv2.putText(img, text, (x0, y0), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #             fontScale=1, color=color, thickness=1)

        return img

    # 1. Load region extraction results (pickle file)
    regions_file = 'log/regions/{}_{}_regions.pkl'.format(args.dataset, args.split)
    with open(regions_file, 'rb') as f:
        obj = pickle.load(f)

    # 2. Define image preprocessing (same as regions_extractor)
    transform = transforms.Compose([lambda x: np.array(x)])

    # 3. Load test set images (visualization uses test set, fixed 'test' split)
    img_name, _, _ = get_dataset_filename('test')
    dataset = ImageDataset(os.path.join(args.data_path, args.dataset), img_name, transform=transform)

    # 4. Iterate over test set images, draw bounding boxes and save
    num_data = len(dataset)
    for i in tqdm(range(num_data)):
        # Load original image
        img = dataset[i]
        # Get current image's region results
        instances = obj[i]["instances"]
        # Draw bounding boxes
        img_with_regions = draw_regions(img, instances)

        # 5. Determine visualization result save path
        ori_img_path = dataset.imgs[i]  # original image path (e.g., images/001.jpg)
        # Replace 'images' in path with 'regions', as visualization result save path
        region_path = os.path.join(dataset.data_path, ori_img_path.replace('images', 'regions'))
        # Ensure save directory exists (e.g., regions directory)
        os.makedirs(os.path.dirname(region_path), exist_ok=True)
        # Save image with bounding boxes drawn
        cv2.imwrite(region_path, img_with_regions)


if __name__ == "__main__":
    """
    Main function: parse command line arguments, choose to execute region extraction or visualization based on arguments
    """
    # 1. Initialize command line argument parser
    parser = argparse.ArgumentParser()

    # 2. Add command line arguments (each argument has default value and description)
    parser.add_argument('--device', default='0', type=str, help='GPU device ID (e.g., 0 for first GPU, multiple GPUs separated by comma)')
    parser.add_argument('--cfg_name', default='X-152', type=str,
                        help='Model name (supports ins_seg/detection/R-50/X-101/X-152)')
    parser.add_argument('--data_path', default='./data', type=str, help='Dataset root path (e.g., ./data)')
    parser.add_argument('--dataset', type=str, default='IAPR-TC',
                        choices=['FLICKR-25K', 'NUS-WIDE', 'IAPR-TC', 'MS-COCO'],
                        help='Dataset name (limited to optional values to avoid input errors)')
    parser.add_argument('--split', default='train', type=str, help='Dataset split (train/test, specify which split to extract regions for)')
    parser.add_argument('--class_thred', type=float, default=0.2, help='Object detection confidence threshold (default 0.2, filter low confidence regions)')
    parser.add_argument('-v', '--visualization', action='store_true', default=False,
                        help='Whether to enable visualization (add -v to execute visualization, otherwise execute region extraction)')

    # 3. Parse command line arguments
    args = parser.parse_args()

    # 4. Set GPU device (based on --device argument)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 5. Branch selection: execute visualization or region extraction
    if args.visualization:
        visualization(args)  # Enable visualization: draw and save bounding box images
    else:
        regions_extractor(args)  # Default: extract object regions and save as pickle file