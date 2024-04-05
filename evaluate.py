import numpy as np
import os, glob
from tqdm import tqdm

class_dict = {
    "car" : 0,
    "person": 1
}

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection
    
    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0
    
    # Calculate intersection area
    intersection_area = w_intersection * h_intersection
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def get_tpfpfn(pred, gt, iou_threshold=0.5):
    """
    Calculate object detection metric given predictions and ground truth.
    """
    num_pred = len(pred)
    num_gt = len(gt)
    tp = 0
    fp = 0
    fn = 0
    
    for pred_box in pred:
        pred_cls, pred_conf, pred_x, pred_y, pred_w, pred_h = pred_box
        pred_box_coords = (pred_x, pred_y, pred_w, pred_h)
        max_iou = 0
        
        for gt_box in gt:
            gt_cls, _, gt_x, gt_y, gt_w, gt_h = gt_box
            gt_box_coords = (gt_x, gt_y, gt_w, gt_h)
            
            if pred_cls == gt_cls:
                iou = calculate_iou(pred_box_coords, gt_box_coords)
                if iou > max_iou:
                    max_iou = iou
        
        if max_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
    
    fn = num_gt - tp
    
    return tp, fp, fn

def get_prec_rec(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

def load_txt(fpath, parser):
    """Parse a detection line and return class_id, confidence, bbox."""
    lines = open(fpath).readlines()
    detections = []
    for line in lines:
        parts = line.strip("\n").split(parser)
        class_id = parts[0]
        confidence = 1.0
        if len(parts) == 6:
            class_id = str(class_dict[class_id])
            confidence = float(parts[1])        
        bbox = [float(x) for x in parts[-4:]]
        detection = tuple([class_id]+[confidence]+bbox)
        detections.append(detection)
    return detections
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser("argument parser")
    parser.add_argument("--pred-dir", type=str,     
                help="It can be a path to image or path to folder of images")
    parser.add_argument("--gt-dir", type=str, 
                help="It can be a path to image or path to folder of images")
    parser.add_argument("--conf", type=float, default=0.5,
                help="confidence threshold")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    run command - 
    python evaluate.py --pred-dir customVisionOutput --gt-dir ../JSON2YOLO/new_dir/labels/val
    """    
    args = parse_args()
    gt_paths = glob.glob(args.gt_dir + "/*.txt")
    pred_fnames = os.listdir(args.pred_dir)
    tps = 0
    fps = 0
    fns = 0
    for gt_path in tqdm(gt_paths):
        fname = os.path.basename(gt_path)
        if fname in pred_fnames:
            gts = load_txt(gt_path, ",")
            preds = load_txt(os.path.join(args.pred_dir, fname), ",")
            tp, fp, fn = get_tpfpfn(preds, gts, iou_threshold=0.5)
            tps += tp
            fps += fp
            fns += fn
    precision, recall = get_prec_rec(tps, fps, fns)

    print("Precision:", precision)
    print("Recall:", recall)
