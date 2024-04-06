import os
import json


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    xA, yA, xB, yB = (
        max(boxA[0], boxB[0]),
        max(boxA[1], boxB[1]),
        min(boxA[2], boxB[2]),
        min(boxA[3], boxB[3]),
    )
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = (
        interArea / float(boxAArea + boxBArea - interArea)
        if boxAArea + boxBArea - interArea > 0
        else 0
    )
    return iou


def load_data(file_path, is_prediction):
    """
    Load data from a file, differentiating between prediction and ground truth files.
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if is_prediction:
                class_name, prob, x_min, y_min, width, height = (
                    parts[0],
                    float(parts[1]),
                    *map(float, parts[2:]),
                )
            else:
                class_name, x_min, y_min, width, height = parts[0], *map(
                    float, parts[1:]
                )
            x_max, y_max = x_min + width, y_min + height
            data.append([class_name, x_min, y_min, x_max, y_max])
    return data


def calculate_metrics(predictions, ground_truths):
    """
    Calculate precision and recall for each class across all predictions and ground truths.
    """
    class_metrics = {}
    # Filter by class
    classes = set(pred[0] for pred in predictions).union(gt[0] for gt in ground_truths)
    # print("pred:", set(pred[0] for pred in predictions))
    # print("gt:", set(gt[0] for gt in ground_truths))

    for cls in classes:
        preds = [p for p in predictions if p[0] == cls]
        gts = [gt for gt in ground_truths if gt[0] == cls]
        tp, fp, fn = 0, 0, len(gts)

        for pred in preds:
            matched = False
            for i, gt in enumerate(gts):
                if calculate_iou(pred[1:], gt[1:]) >= 0.5:
                    tp += 1
                    fn -= 1
                    matched = True
                    break
            if not matched:
                fp += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        class_metrics[cls] = (precision, recall)

    return class_metrics


def evaluate_performance(pred_dir, gt_dir):
    """
    Evaluate performance, producing per-class average precision, recall, and F1 score across all file pairs.
    """
    pred_files = sorted(
        [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
    )
    gt_files = sorted(
        [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
    )

    aggregate_metrics = {}

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path, gt_path = os.path.join(pred_dir, pred_file), os.path.join(
            gt_dir, gt_file
        )
        predictions, ground_truths = load_data(pred_path, True), load_data(
            gt_path, False
        )
        metrics = calculate_metrics(predictions, ground_truths)

        for cls, metrics in metrics.items():
            precision, recall = metrics
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            if cls not in aggregate_metrics:
                aggregate_metrics[cls] = {"precision": [], "recall": [], "f1": []}
            aggregate_metrics[cls]["precision"].append(precision)
            aggregate_metrics[cls]["recall"].append(recall)
            aggregate_metrics[cls]["f1"].append(f1_score)

    for cls, metrics in aggregate_metrics.items():
        avg_precision = sum(metrics["precision"]) / len(metrics["precision"])
        avg_recall = sum(metrics["recall"]) / len(metrics["recall"])
        avg_f1 = sum(metrics["f1"]) / len(metrics["f1"])
        print(
            f"Class: {cls}, Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, Average F1: {avg_f1:.4f}"
        )


if __name__ == "__main__":
    pred_dir_path = "/Users/rishabpal/Downloads/rishabh/imageAnalysisOutput"
    gt_dir_path = "/Users/rishabpal/Downloads/rishabh/coco_annotations/output/txt_files"
    evaluate_performance(pred_dir_path, gt_dir_path)
