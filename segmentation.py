from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

def run_segmentation(image):
    results = model(image)
    if results[0].masks is None:
        return None
    return results[0].masks.data.cpu().numpy()