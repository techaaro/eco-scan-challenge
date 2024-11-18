from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Load video
vid = cv2.VideoCapture('video.mp4')
vid.set(3, 640)
vid.set(4, 480)
model = YOLO("Model and Results/best.pt")

# Class names and carbon footprint values (kg CO₂e per item)
classNames = [
    "Blazer", "Blouse", "Cap", "Hoodie", "Jacket", "Jumpsuit",
    "Longsleeve", "Onepiece", "Other", "Pants", "Shirt", "Shoes",
    "Shorts", "Skirt", "T-Shirt", "Top", "Underwear", "Vest"
]
carbon_footprint = {
    "Blazer": 25,
    "Blouse": 10,
    "Cap": 3,
    "Hoodie": 15,
    "Jacket": 25,
    "Jumpsuit": 20,
    "Longsleeve": 8,
    "Onepiece": 18,
    "Other": 5,
    "Pants": 40,
    "Shirt": 8,
    "Shoes": 20,
    "Shorts": 6,
    "Skirt": 10,
    "T-Shirt": 7,
    "Top": 5,
    "Underwear": 2,
    "Vest": 10
}

# Define video writer object for saving output
output_size = (400, 600)  # Desired output size (width, height)
output_fps = int(vid.get(cv2.CAP_PROP_FPS))  # Get FPS from input video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter('output.mp4', fourcc, output_fps, output_size)

# Initialize total carbon footprint
total_carbon_footprint = 0

while vid.isOpened():
    success, frame = vid.read()
    if not success:
        break

    results = model(frame, stream=True)
    frame_carbon_footprint = 0  # For the current frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(frame, (x1, y1, w, h), (255, 0, 0))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            if cls < len(classNames):  # Avoid out-of-range errors
                class_name = classNames[cls]
                item_footprint = carbon_footprint.get(class_name, 0)
                frame_carbon_footprint += item_footprint

                # Display class name and confidence
                cvzone.putTextRect(
                    frame,
                    f'{class_name} {conf} ({item_footprint} kg CO2e)',
                    (max(0, x1), max(35, y1)),
                    scale=3,
                    thickness=3
                )

    # Update and display total carbon footprint
    total_carbon_footprint += frame_carbon_footprint
    cvzone.putTextRect(
        frame,
        f'Footprint: {total_carbon_footprint:.2f} kg CO2e',
        (50, 100),
        scale=5,
        thickness=4,
        colorR=(0, 255, 0)
    )

    # Resize frame and write to output video
    resized_frame = cv2.resize(frame, output_size)
    out.write(resized_frame)

    # Display the video frame
    cv2.imshow("Image", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Release resources
vid.release()
out.release()
cv2.destroyAllWindows()
