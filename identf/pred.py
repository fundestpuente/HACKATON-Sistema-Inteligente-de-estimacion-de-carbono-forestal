import torch
from torchvision import transforms
from PIL import Image
from resnet import getresnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("resnet18_arboles_ec.pth", map_location=device)
classes = checkpoint["classes"]

model = getresnet18(len(classes))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]
