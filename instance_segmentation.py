import cv2
import numpy as np
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device) # Instance segmentation model

video_path = "/home/prasad-nirmal/Documents/FYP/Videos/camp.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 480))
    results = model(resized_frame)

    masked_frame = resized_frame.copy()
    masks = results[0].masks

    if masks is not None:
        for mask in masks.data:
            mask = mask.cpu().numpy().astype(np.uint8) * 255

            mask_resized = cv2.resize(mask, (resized_frame.shape[1], resized_frame.shape[0]))

            colored_mask = np.zeros_like(resized_frame)
            colored_mask[:, :, 1] = mask_resized  # Green mask

            masked_frame = cv2.addWeighted(masked_frame, 1.0, colored_mask, 0.5, 0)

    cv2.imshow("YOLOv8 Segmentation Mask Only", masked_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
