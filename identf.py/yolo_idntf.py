from ultralytics import YOLO
from PIL import Image
from ultralytics import settings

print(settings)

settings.update({"runs_dir": "/path/to/runs"})


settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

settings.reset()

yolo_model = YOLO("yolov8n.pt")  

def detect_tree(image_path, conf=0.25):
   
    results = yolo_model(image_path, conf=conf)
    img = Image.open(image_path).convert("RGB")

    if len(results[0].boxes) == 0:
        return img  

    # Tomar la caja más grande
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
    best_box = boxes[areas.index(max(areas))]

    x1, y1, x2, y2 = map(int, best_box)
    crop = img.crop((x1, y1, x2, y2))

    return crop


