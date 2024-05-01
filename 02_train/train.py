from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='/mnt/c/Users/Public/RWC', epochs=20, imgsz=64)