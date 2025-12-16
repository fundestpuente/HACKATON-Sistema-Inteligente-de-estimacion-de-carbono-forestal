import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet import getresnet18

#no creo que mi dispositivo sea compatible con cuda
device = "cuda" if torch.cuda.is_available() else "cpu"

#data
transform = transforms.Compose([
    transforms.Resize((224, 224)), #solo con esto puede trabajar resnet
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    "ruta/dataset/nodefinidaxd", #crear ruta
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

num_classes = len(train_dataset.classes)

#modelo
model = getresnet18(num_classes)
model.to(device)

#entrena
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(15):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {running_loss / len(train_loader):.4f}")

#se guarda
torch.save(model.state_dict(), "resnet18_trees.pth")
print(" Modelo guardado")
