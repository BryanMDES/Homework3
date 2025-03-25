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


BATCH_SIZE = 16
LR = 0.0003
EPOCHS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = torch.tensor([0.1, 2.5, 2.5]).to(DEVICE)  # Reduce background weight
seg_loss_fn = nn.CrossEntropyLoss(weight=class_weights) # segmentation loss
depth_loss_fn = nn.L1Loss()  # Depth loss

# Measuing how well two areas overlap
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    target = F.one_hot(target.long(), num_classes=pred.shape[1]) 
    target = target.permute(0, 3, 1, 2).float()  
    
    intersection = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Another overlap measuring
def iou_loss(pred, target, smooth=1e-6):
  pred = torch.sigmoid(pred)

  target = F.one_hot(target.long(), num_classes=pred.shape[1])  
  target = target.permute(0, 3, 1, 2).float()  

  intersection = (pred * target).sum((1, 2))
  union = (pred + target - pred * target).sum((1, 2))
  iou = (intersection + smooth) / (union + smooth)

  return (1 - iou.mean()) 

# Logs being saved in default as logs and making training results the same everytime sa default 2024
def train(exp_dir: str = "logs", seed: int = 2024):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed) #Makes training repetaable

    
    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    
    # Loading the training data
    train_loader = load_data(
    dataset_path="/content/Homework3/homework3/drive_data/train",
    transform_pipeline="default",  
    batch_size=BATCH_SIZE,
    shuffle=True
)

    # Loading the validation data
    val_loader = load_data(
    dataset_path="/content/Homework3/homework3/drive_data/val",
    transform_pipeline="default",  
    batch_size=BATCH_SIZE,
    shuffle=False
)

    
    model = Detector().to(device) #Adamm is popular that adjusts the learning rate automatically

    optimizer = optim.Adam(model.parameters(), lr=LR) # Sets the leraning rate

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
      
        total_seg_loss, total_depth_loss, total_iou_loss = 0.0, 0.0, 0.0

        first_batch = True

        for batch in train_loader:
            images = batch["image"].to(device)  # Inputs pictures to the device
            depth = batch["depth"].to(device)  # How far things are in the picture
            track = batch["track"].to(device)  # What each pixel is 

            optimizer.zero_grad() #Erasing old learning notes

            pred_seg, pred_depth = model(images) # Guestting what each pixel is and how far each pixel is

            # Calculating how wrong it s
            loss_seg = 0.5 * seg_loss_fn(pred_seg, track) + 0.5 * dice_loss(pred_seg, track) 
            loss_depth = depth_loss_fn(pred_depth, depth)  
            loss_iou = iou_loss(pred_seg, track)  

            
            # Give different weights to teh losses, segmentation is most important, the IoU, then the depth
            loss = 1.5 * loss_seg + 0.5 * loss_depth + 0.8 * loss_iou  
            loss.backward() # How to fix the model
            optimizer.step() 

            total_seg_loss += loss_seg.item() #Adds this batch loss to the total so we can compute averages and track the progress
            total_depth_loss += loss_depth.item()
            total_iou_loss += loss_iou.item()
            global_step += 1
            # Sending loss values to the tensor
            logger.add_scalar("train/seg_loss", loss_seg.item(), global_step)
            logger.add_scalar("train/depth_loss", loss_depth.item(), global_step)
            logger.add_scalar("train/iou_loss", loss_iou.item(), global_step)
            logger.flush() 
        
        with torch.inference_mode(): #turns off the gradient tracking faster with less memory
            model.eval() #Putting the model in eval mode
            val_seg_loss, val_depth_loss, val_iou_loss = 0, 0, 0 #How well the model does
            mean_iou = 0

            for batch in val_loader:
                images = batch["image"].to(device) #New test images
                depth = batch["depth"].to(device) 
                track = batch["track"].to(device) #Correct labels
                pred_seg, pred_depth = model(images) # Predicted class for each pixel
                loss_seg =0.5 * seg_loss_fn(pred_seg, track) + 0.5 * dice_loss(pred_seg, track) #How wrong teh model is
                loss_depth = depth_loss_fn(pred_depth, depth)
                loss_iou = iou_loss(pred_seg, track)
                val_seg_loss += loss_seg.item()
                val_depth_loss += loss_depth.item()
                val_iou_loss += loss_iou.item()

                mean_iou += loss_iou.item()

        mean_iou /= len(val_loader) #Giving me the iou acccorss all batches

        logger.add_scalar("val/seg_loss", val_seg_loss / len(val_loader), global_step)
        logger.add_scalar("val/depth_loss", val_depth_loss / len(val_loader), global_step)
        logger.add_scalar("val/iou_loss", val_iou_loss / len(val_loader), global_step)  
        logger.flush()

        print(f"Epoch {epoch+1}/{EPOCHS} - Seg Loss: {total_seg_loss:.4f}, Depth Loss: {total_depth_loss:.4f}")
        print(f"             - val SEG LOSS: {val_seg_loss: .4f}, Val Deph Loss: {val_depth_loss: .4f}")
    
    save_model(model)
    print("detector.th Saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))