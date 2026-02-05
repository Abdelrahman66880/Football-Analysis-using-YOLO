from ultralytics import YOLO

model = YOLO("models/best.pt")
result = model.predict(source="input_videos/Video_1.mp4",save=True)
print(result[0])
print("===============================")
for box in result[0].boxes:
    print(box)
    