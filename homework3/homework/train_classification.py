import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Classifier  # Import your CNN model
from datasets.classification_dataset import load_data  # Adjust path if needed

# 1️⃣ **Set device (GPU if available)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ **Load the Dataset**
train_loader = load_data("/content/Homework3/homework3/classification_data/train", batch_size=32)
val_loader = load_data("/content/Homework3/homework3/classification_data/val", batch_size=32)

# 3️⃣ **Initialize Model, Loss, and Optimizer**
model = Classifier().to(device)  # Use your CNN model
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# 4️⃣ **Training Loop**
num_epochs = 10  # You can change this
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

# 5️⃣ **Save the Model**
torch.save(model.state_dict(), "classifier.pth")
print("Model saved as classifier.pth")
