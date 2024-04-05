import numpy as np

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

def calculate_detection_metric(pred, gt, iou_threshold=0.5):
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
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

# Example usage
pred = [
    ('car', 0.83984375, 0.24296875, 0.47158403869407495, 0.0953125, 0.1003627569528416),
    ('person', 0.8271484375, 0.6203125, 0.4619105199516324, 0.02734375, 0.08101571946795647)
]
gt = [
    ('car', 0.83984375, 0.24296875, 0.47158403869407495, 0.0953125, 0.1003627569528416),
    ('person', 0.8271484375, 0.6203125, 0.4619105199516324, 0.02734375, 0.08101571946795647)
]
precision, recall = calculate_detection_metric(pred, gt)
print("Precision:", precision)
print("Recall:", recall)
