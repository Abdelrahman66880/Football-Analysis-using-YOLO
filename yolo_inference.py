from ultralytics import YOLO

model = YOLO("yolov8x")
result = model.predict(source="input_videos/Video_1.mp4",save=True)
print(result[0])
print("===============================")
for box in result[0].boxes:
    print(box)
    