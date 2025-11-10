import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse
from prettytable import PrettyTable

def get_arguments():
    parser = argparse.ArgumentParser(description="PerSAM2 Evaluation (mIoU & bIoU)")
    parser.add_argument('--gt_root', type=str, default='./data/Annotations',
                        help='Root directory containing Ground Truth class folders')
    parser.add_argument('--pred_root', type=str, default='./outputs',
                        help='Root directory containing Prediction class folders')
    return parser.parse_args()

def mask_iou(pred, target):
    """
    Calculate standard Intersection over Union (IoU) for binary masks.
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 # Both empty is perfect match
    return intersection / union

def mask_biou(pred, target, dilation_pixels=5):
    """
    Calculate Boundary IoU (bIoU).
    """
    # Get boundaries by dilating and eroding to find the contours
    kernel = np.ones((3, 3), np.uint8)
    
    # Define Boundary of Prediction
    pred_dilated = cv2.dilate(pred.astype(np.uint8), kernel, iterations=dilation_pixels)
    pred_eroded = cv2.erode(pred.astype(np.uint8), kernel, iterations=dilation_pixels)
    pred_boundary = pred_dilated - pred_eroded

    # Define Boundary of Ground Truth
    target_dilated = cv2.dilate(target.astype(np.uint8), kernel, iterations=dilation_pixels)
    target_eroded = cv2.erode(target.astype(np.uint8), kernel, iterations=dilation_pixels)
    target_boundary = target_dilated - target_eroded

    # Calculate IoU of these boundaries
    intersection = np.logical_and(pred_boundary > 0, target_boundary > 0).sum()
    union = np.logical_or(pred_boundary > 0, target_boundary > 0).sum()
    
    if union == 0:
        return 1.0
    return intersection / union

def evaluate_class(class_name, gt_class_dir, pred_class_dir):
    """
    Evaluates all matching images in a class directory.
    """
    pred_paths = sorted(glob.glob(os.path.join(pred_class_dir, "*.png")))
    
    ious = []
    bious = []

    for pred_path in pred_paths:
        file_name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_class_dir, file_name)

        if not os.path.exists(gt_path):
            # Skip if no corresponding GT (e.g. visualizations saved as .jpg)
            continue

        # Load Masks
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Binarize (ensure they are 0 or 1)
        pred_mask = (pred_mask > 127).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        # Calculate Metrics
        ious.append(mask_iou(pred_mask, gt_mask))
        bious.append(mask_biou(pred_mask, gt_mask))

    if len(ious) == 0:
        return 0.0, 0.0

    return np.mean(ious) * 100, np.mean(bious) * 100

def main():
    args = get_arguments()

    if not os.path.exists(args.pred_root):
        print(f"Error: Prediction root not found at {args.pred_root}")
        return

    # Find all classes based on subdirectories in pred_root
    classes = sorted([d for d in os.listdir(args.pred_root) if os.path.isdir(os.path.join(args.pred_root, d))])
    
    print(f"Found {len(classes)} classes to evaluate.")

    table = PrettyTable()
    table.field_names = ["Class Name", "mIoU (%)", "bIoU (%)"]
    table.align["Class Name"] = "l" 

    all_class_miou = []
    all_class_biou = []

    for class_name in tqdm(classes, desc="Evaluating Classes"):
        pred_class_dir = os.path.join(args.pred_root, class_name)
        gt_class_dir = os.path.join(args.gt_root, class_name)

        if not os.path.exists(gt_class_dir):
            print(f"Warning: GT missing for class '{class_name}', skipping.")
            continue

        miou, biou = evaluate_class(class_name, gt_class_dir, pred_class_dir)
        
        all_class_miou.append(miou)
        all_class_biou.append(biou)
        table.add_row([class_name, f"{miou:.2f}", f"{biou:.2f}"])

    # Calculate Final Averages
    final_miou = np.mean(all_class_miou) if all_class_miou else 0.0
    final_biou = np.mean(all_class_biou) if all_class_biou else 0.0

    print("\n====== PerSAM2 Evaluation Results ======")
    print(table)
    print("\n====== Final Averages ======")
    print(f"Mean IoU (mIoU): {final_miou:.2f}%")
    print(f"Boundary IoU (bIoU): {final_biou:.2f}%")

if __name__ == '__main__':
    main()