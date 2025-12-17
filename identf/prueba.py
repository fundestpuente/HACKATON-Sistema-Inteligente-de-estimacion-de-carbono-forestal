import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet import getresnet18

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_res")


device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

checkpoint = torch.load("resnet18_arboles_ec.pth", map_location=device)
classes = checkpoint["classes"]

model = getresnet18(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Accuracy en test: {100 * correct / total:.2f}%")
