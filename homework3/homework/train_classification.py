import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Classifier 
from datasets.classification_dataset import load_data  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = load_data(
    dataset_path="/content/Homework3/homework3/classification_data/train",
    transform_pipeline="aug",  # Loading images in batches of 32 and random image changes
    batch_size=32,
    shuffle=True
)

val_loader = load_data(
    dataset_path="/content/Homework3/homework3/classification_data/val",
    transform_pipeline="default",  # Load validation images in batches of 32, no random changes here because we are using this to test on how good th model is
    batch_size=32,
    shuffle=False
)

model = Classifier().to(device)  # Model will learn to clasify images
criterion = nn.CrossEntropyLoss()  # How wrong it is
optimizer = optim.Adam(model.parameters(), lr=0.001)  # updateing the model to get better at each step

num_epochs = 10  
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0 #Keeping track of accuracy or the loss

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Clearing the old gradients
        outputs = model(images) # Making a prediction
        loss = criterion(outputs, labels) # Calculating the inaccuracy
        loss.backward()
        optimizer.step() #Updating model weights
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}")

torch.save(model.state_dict(), "classifier.th")
print("classifier.th Saved")
