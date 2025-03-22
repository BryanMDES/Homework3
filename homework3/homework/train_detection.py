import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from datasets.road_dataset import load_data  
from models import Detector, save_model  


BATCH_SIZE = 16
LR = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    
    seg_loss_fn = nn.CrossEntropyLoss()  
    depth_loss_fn = nn.L1Loss()  

    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    global_step = 0

    
    for epoch in range(EPOCHS):
        model.train()
        total_seg_loss, total_depth_loss = 0, 0

        for batch in train_loader:
            images = batch["image"].to(device)  
            depth = batch["depth"].to(device)  
            track = batch["track"].to(device)  

            optimizer.zero_grad()

            
            pred_seg, pred_depth = model(images)

            
            loss1 = seg_loss_fn(pred_seg, track)  
            loss2 = depth_loss_fn(pred_depth, depth)  
            loss = loss1 + loss2  

            
            loss.backward()
            optimizer.step()

            total_seg_loss += loss1.item()
            total_depth_loss += loss2.item()

            global_step += 1

            logger.add_scalar("train/seg_loss", loss1.item(), global_step)
            logger.add_scalar("train/depth_loss", loss2.item(), global_step)
            logger.flush()

        
        with torch.inference_mode():
            model.eval()
            val_seg_loss, val_depth_loss = 0, 0

            for batch in val_loader:
                images = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                pred_seg, pred_depth = model(images)

                loss1 = seg_loss_fn(pred_seg, track)
                loss2 = depth_loss_fn(pred_depth, depth)

                val_seg_loss += loss1.item()/len(val_loader)
                val_depth_loss += loss2.item()/len(val_loader)

        
        logger.add_scalar("val/seg_loss", val_seg_loss / len(val_loader), global_step)
        logger.add_scalar("val/depth_loss", val_depth_loss / len(val_loader), global_step)
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