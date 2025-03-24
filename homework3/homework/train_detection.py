import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
import torch.nn.functional as F

from datasets.road_dataset import load_data  
from models import Detector, save_model
from iou_visualizer import handle_bboxes  


BATCH_SIZE = 16
LR = 0.0005
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = torch.tensor([0.1, 2, 2]).to(DEVICE)  # Reduce background weight
seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
depth_loss_fn = nn.L1Loss() 

def iou_loss(pred, target, smooth=1e-6):
  pred = torch.softmax(pred, dim=1)

  target = F.one_hot(target.long(), num_classes=pred.shape[1])  # Ensure int64 type
  target = target.permute(0, 3, 1, 2).float()  # Change shape to (B, C, H, W)

  intersection = (pred * target).sum((1, 2))
  union = (pred + target - pred * target).sum((1, 2))
  iou = (intersection + smooth) / (union + smooth)

  return (1 - iou.mean())  # Higher IoU is better, so we subtract from 1

def compute_iou(pred, target, num_classes=3, smooth=1e-6):
    """
    Computes IoU for each class and returns the mean IoU.
    """
    pred = pred.argmax(dim=1)  # Convert logits to class predictions
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls).float()
        target_mask = (target == cls).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
    
    return sum(ious) / len(ious)  # Mean IoU

def train(exp_dir: str = "logs", seed: int = 2024):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)

    
    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    

    train_loader = load_data(
    dataset_path="/content/Homework3/homework3/drive_data/train",
    transform_pipeline="default",  
    batch_size=BATCH_SIZE,
    shuffle=True
)

    val_loader = load_data(
    dataset_path="/content/Homework3/homework3/drive_data/val",
    transform_pipeline="default",  
    batch_size=BATCH_SIZE,
    shuffle=False
)


    
    model = Detector().to(device)
    #seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights) # Replace CrossEntropyLoss
    #depth_loss_fn = nn.L1Loss()  

    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    global_step = 0

    
    for epoch in range(EPOCHS):
        model.train()
        total_seg_loss, total_depth_loss, total_iou_loss = 0, 0, 0

        first_batch = True

        for batch in train_loader:
            images = batch["image"].to(device)  
            depth = batch["depth"].to(device)  
            track = batch["track"].to(device)  

            optimizer.zero_grad()

            pred_seg, pred_depth = model(images)

            loss_seg = seg_loss_fn(pred_seg, track)  
            loss_depth = depth_loss_fn(pred_depth, depth)  
            loss_iou = iou_loss(pred_seg, track)  

            
            loss = 1.5 * loss_seg + 0.5 * loss_depth + 0.3 * loss_iou  
            loss.backward()
            optimizer.step()

            if first_batch:
              handle_bboxes(images, track, pred_seg, loss_iou.item(), epoch, logger, log_dir)
              first_batch = False  # Ensure it's only visualized once per epoch

            total_seg_loss += loss_seg.item()
            total_depth_loss += loss_depth.item()
            total_iou_loss += loss_iou.item()

            global_step += 1

            logger.add_scalar("train/seg_loss", loss_seg.item(), global_step)
            logger.add_scalar("train/depth_loss", loss_depth.item(), global_step)
            logger.add_scalar("train/iou_loss", loss_iou.item(), global_step)
            logger.flush()
        
        with torch.inference_mode():
            model.eval()
            val_seg_loss, val_depth_loss, val_iou_loss = 0, 0, 0
            mean_iou = 0

            for batch in val_loader:
                images = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                pred_seg, pred_depth = model(images)

                loss_seg = seg_loss_fn(pred_seg, track)
                loss_depth = depth_loss_fn(pred_depth, depth)
                loss_iou = iou_loss(pred_seg, track)

                val_seg_loss += loss_seg.item()
                val_depth_loss += loss_depth.item()
                val_iou_loss += loss_iou.item()

                mean_iou += compute_iou(pred_seg,track).item()

        mean_iou /= len(val_loader)

        logger.add_scalar("val/seg_loss", val_seg_loss / len(val_loader), global_step)
        logger.add_scalar("val/depth_loss", val_depth_loss / len(val_loader), global_step)
        logger.add_scalar("val/iou_loss", val_iou_loss / len(val_loader), global_step)  # Log IoU validation loss
        logger.flush()

        print(f"Epoch {epoch+1}/{EPOCHS} - Seg Loss: {total_seg_loss:.4f}, Depth Loss: {total_depth_loss:.4f}")
        print(f"             - val SEG LOSS: {val_seg_loss: .4f}, Val Deph Loss: {val_depth_loss: .4f}")

    
    save_model(model)
    print("Model saved as detector.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))