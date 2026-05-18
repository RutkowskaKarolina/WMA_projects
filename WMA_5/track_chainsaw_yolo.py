from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.track(
    source="materials/sawmovie.mp4",
    conf=0.3,
    iou=0.5,
    show=True,
    save=True,
    persist=True
)
