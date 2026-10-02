import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

def fine_tune_model():
    # 1. Device configuration (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Augmentation and Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = './data/raw'
    if not os.path.exists(data_dir):
        print(f"Error: Local dataset folder '{data_dir}' not found!")
        return

    # Load dataset from class folders
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {len(full_dataset)} images across {num_classes} classes.")
    print(f"Classes: {class_names}")

    # 3. Split into 80% Training and 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    }

    # 4. Load Pre-trained ResNet18 and Modify Classifier
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze feature extraction layers, train only the classifier head
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 5. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 6. Training Loop
    num_epochs = 5
    print("\nStarting fine-tuning...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    # 7. Save the fine-tuned model weights locally
    os.makedirs('./models', exist_ok=True)
    save_path = './models/plant_disease_resnet18.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned model successfully saved to {save_path}!")

if __name__ == '__main__':
    fine_tune_model()