import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import gc

# cleaned up version of the code
# copilot helped me to clean up the code
# chatgpt helped me figure out how to do this.
# I have added the comments to explain the code

def extract_bounding_boxes(mask, scale_x, scale_y):
    """
    Extracts separate bounding boxes **without OpenCV**.
    
    Args:
        mask (torch.Tensor): Binary mask for a specific class (H, W).
        scale_x (float): Scaling factor for x-coordinates.
        scale_y (float): Scaling factor for y-coordinates.

    Returns:
        list of bounding boxes [(x_min, y_min, x_max, y_max)].
    """

    visited = torch.zeros_like(mask)  # Keep track of visited pixels
    bboxes = []

    # Iterate through the mask to find individual connected components
    for y, x in zip(*torch.where(mask > 0)):  # Get all non-zero points
        if visited[y, x]:  # Skip if already visited
            continue

        # Extract all connected pixels for the current object
        object_mask = mask.clone()
        object_mask[mask != mask[y, x]] = 0  # Keep only current object's pixels
        object_coords = torch.where(object_mask > 0)

        if len(object_coords[0]) > 0:  # Ensure there is a valid region
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
    Extracts, scales, and logs **separate** bounding boxes from segmentation masks without OpenCV.

    Args:
        img (torch.Tensor): The input image tensor (1, C, H, W).
        track_mask (torch.Tensor): Ground truth segmentation mask (1, H, W).
        pred_classes (torch.Tensor): Predicted segmentation mask (1, H, W).
        iou (float): Intersection over Union value.
        epoch (int): Current epoch number.
        logger (tb.SummaryWriter): TensorBoard logger.
        log_dir (Path): Directory to save images.
    """
    os.makedirs(str(log_dir), exist_ok=True)
    batch_image = img[0].cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) -> (H, W, C)

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

        # ✅ **Apply bounding box extraction without OpenCV**
        gt_bboxes.extend(extract_bounding_boxes(gt_mask, scale_x, scale_y))
        pred_bboxes.extend(extract_bounding_boxes(pred_mask, scale_x, scale_y))

        # Assign IoU per detected object
        iou_values.extend([iou] * len(pred_bboxes))

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
        iou_values = torch.tensor(iou_values, dtype=torch.float32)

        #print(f"DEBUG - GT BBoxes: {gt_bboxes.shape}, Pred BBoxes: {pred_bboxes.shape}, IoUs: {iou_values.shape}")

        output_path = os.path.join(str(log_dir), f"epoch_{epoch+1}_iou.png")  # Ensure correct path format
        print(f"✅ DEBUG: Saving IoU visualization at: {output_path}")  # Debugging print

        #os.makedirs(str(log_dir), exist_ok=True)

        draw_iou_bounding_boxes(batch_image, gt_bboxes, pred_bboxes, iou_values, str(output_path))

        logger.add_image("Validation/BoundingBoxes", np.array(Image.open(output_path)), epoch, dataformats='HWC')
        
        print(f"✅ DEBUG: handle_bboxes() called for epoch {epoch+1}")  # Debugging print
        print(f"DEBUG: Saving IoU visualization at: {log_dir / f'epoch_{epoch+1}_iou.png'}")


def draw_iou_bounding_boxes(image, gt_bboxes, pred_bboxes, ious, output_path="output.png"):
    """
    Draws ground truth (red) and predicted (yellow) bounding boxes with IoU values.

    Args:
        image (torch.Tensor or np.ndarray): Input image (C, H, W) or (H, W, C).
        gt_bboxes (torch.Tensor): Ground truth bounding boxes [N, 4] -> [x_min, y_min, x_max, y_max].
        pred_bboxes (torch.Tensor): Predicted bounding boxes [N, 4] -> [x_min, y_min, x_max, y_max].
        ious (torch.Tensor): IoU values of shape (N,).
        output_path (str): Path to save the rendered image.
    """
    print(f"✅ DEBUG: Saving IoU visualization at {output_path}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ DEBUG: Output directory confirmed at {output_dir}")

  
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy() if image.dim() == 3 else image.cpu().numpy()

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Define constant colors
    color_gt = "#FF0000"  # Red for ground truth
    color_pred = "#FFFF00"  # Yellow for predictions

    # Draw ground truth boxes first (Red)
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color_gt, facecolor="none")
        ax.add_patch(rect)

     # Debug: Print bounding box details
    print(f"Ground Truth BBoxes: {gt_bboxes}")
    print(f"Predicted BBoxes: {pred_bboxes}")
    print(f"IoU Values: {ious}")

    # Draw predicted boxes on top (Yellow)
    for bbox, iou in zip(pred_bboxes, ious):
        x_min, y_min, x_max, y_max = bbox.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color_pred, facecolor="none", linestyle="dashed")
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"IoU: {iou:.2f}", color="black", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Save a dummy image to test saving
    debug_image = np.random.rand(128, 128, 3)  # Generate a random image
    plt.imsave(output_path.replace(".png", "_debug.png"), debug_image)  # Save debug image


    if os.path.exists(output_path):
        print(f"✅ Image successfully saved at {output_path}")
    else:
        print(f"❌ Image saving failed at {output_path}")

    print(f"Visualization saved at: {output_path}")
    #gc.collect()