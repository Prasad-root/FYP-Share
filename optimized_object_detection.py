import cv2
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

video_path = "/home/prasad-nirmal/Documents/FYP/Videos/camp.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
save_output = True

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 480))

    if frame_count % 2 == 0:
        results = model(resized_frame)
        annotated_frame = results[0].plot()

        
        if save_output and out is None:
            height, width = annotated_frame.shape[:2]
            out = cv2.VideoWriter("output_detected.mp4", fourcc, 20.0, (width, height))

        if save_output:
            out.write(annotated_frame)
        
        cv2.imshow("YOLOv8 - Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
