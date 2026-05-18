from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="chainsaw_dataset/data.yaml", epochs=50)