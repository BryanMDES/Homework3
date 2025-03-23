import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging

# ✅ Initialize logging to track issues
log_file = "/content/Homework3/homework3/logs/iou_visualizer.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure logs directory exists
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_bounding_boxes(mask, scale_x, scale_y):
    """
    Extracts separate bounding boxes **without OpenCV**.
    """
    visited = torch.zeros_like(mask)  # Keep track of visited pixels
    bboxes = []

    for y, x in zip(*torch.where(mask > 0)):  # Get all non-zero points
        if visited[y, x]:  # Skip if already visited
            continue

        object_mask = mask.clone()
        object_mask[mask != mask[y, x]] = 0
        object_coords = torch.where(object_mask > 0)

        if len(object_coords[0]) > 0:
            x_min, x_max = object_coords[1].min().item(), object_coords[1].max().item()
            y_min, y_max = object_coords[0].min().item(), object_coords[0].max().item()
            
            # Scale bounding box coordinates
            x_min, x_max = int(x_min * scale_x), int(x_max * scale_x)
            y_min, y_max = int(y_min * scale_y), int(y_max * scale_y)

            # Store the bounding box
            bboxes.append([x_min, y_min, x_max, y_max])

            # Mark region as visited
            visited[object_coords] = 1

    return bboxes


def handle_bboxes(img, track_mask, pred_classes, iou, epoch, logger, log_dir):
    """
    Extracts, scales, and logs bounding boxes from segmentation masks.
    """
    os.makedirs(str(log_dir), exist_ok=True)  # ✅ Ensure log directory exists

    batch_image = img[0].cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)

    if batch_image.max() <= 1.0:
        batch_image = (batch_image * 255).astype(np.uint8)

    orig_h, orig_w = batch_image.shape[:2]
    mask_h, mask_w = track_mask.shape[1:]

    scale_x = orig_w / mask_w
    scale_y = orig_h / mask_h

    gt_bboxes = []
    pred_bboxes = []
    iou_values = []

    for idx in torch.unique(track_mask):
        if idx == 0:
            continue  # Skip background class

        gt_mask = (track_mask[0] == idx).float()
        pred_mask = (pred_classes[0] == idx).float()

        gt_bboxes.extend(extract_bounding_boxes(gt_mask, scale_x, scale_y))
        pred_bboxes.extend(extract_bounding_boxes(pred_mask, scale_x, scale_y))
        iou_values.extend([iou] * len(pred_bboxes))

    output_path = os.path.join(str(log_dir), f"epoch_{epoch+1}_iou.png")
    logging.info(f"✅ Epoch {epoch+1}: Preparing to save IoU visualization at {output_path}")

    # ✅ Save a debug test image
    debug_test_image = np.random.rand(128, 128, 3)
    debug_path = output_path.replace(".png", "_debug.png")
    plt.imsave(debug_path, debug_test_image)

    if os.path.exists(debug_path):
        logging.info(f"✅ Debug test image successfully saved at {debug_path}")
    else:
        logging.error(f"❌ Debug test image saving failed at {debug_path}")

    # Call drawing function
    draw_iou_bounding_boxes(batch_image, gt_bboxes, pred_bboxes, iou_values, output_path)

    if os.path.exists(output_path):
        logging.info(f"✅ Final check: IoU Image successfully saved at {output_path}")
    else:
        logging.error(f"❌ Final check: IoU Image is still missing at {output_path}")

    # Log the image in TensorBoard
    logger.add_image("Validation/BoundingBoxes", np.array(Image.open(output_path)), epoch, dataformats='HWC')


def draw_iou_bounding_boxes(image, gt_bboxes, pred_bboxes, ious, output_path="output.png"):
    """
    Draws ground truth (red) and predicted (yellow) bounding boxes with IoU values.
    """
    logging.info(f"🔍 Attempting to save IoU visualization at {output_path}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    color_gt = "#FF0000"  # Red for ground truth
    color_pred = "#FFFF00"  # Yellow for predictions

    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = np.array(bbox)
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color_gt, facecolor="none")
        ax.add_patch(rect)

    for bbox, iou in zip(pred_bboxes, ious):
        x_min, y_min, x_max, y_max = np.array(bbox)
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color_pred, facecolor="none", linestyle="dashed")
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"IoU: {iou:.2f}", color="black", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    if os.path.exists(output_path):
        logging.info(f"✅ Image successfully saved at {output_path}")
    else:
        logging.error(f"❌ Image saving failed at {output_path}")

    # ✅ Save a debug image to confirm writing works
    debug_image = np.random.rand(128, 128, 3)
    plt.imsave(output_path.replace(".png", "_debug.png"), debug_image)
