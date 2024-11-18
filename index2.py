import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

# Load image
image_path = 'Test\p21.jpg'  # Path to the input image
# input_size = (800, 600)
output_path = 'output_image.jpg'  # Path to save the output image
image = cv2.imread(image_path)
# image = cv2.resize(image, input_size)

# Load YOLO model
model = YOLO("Model and Results/best1.pt")

# Class names and carbon footprint values (kg CO₂e per item)
classNames = ['dress', 'jacket', 'pants', 'polo', 'shirt', 'shoes', 'shorts', 'tshirt']
carbon_footprint = {
    "dress": 20,       # Example: Cotton dress
    "jacket": 25,      # Example: Polyester jacket
    "pants": 40,       # Example: Denim pants
    "polo": 10,        # Example: Cotton polo shirt
    "shirt": 8,        # Example: Casual shirt
    "shoes": 20,       # Example: Leather shoes
    "shorts": 6,       # Example: Cotton shorts
    "tshirt": 7
}

# Process the image
results = model(image)
total_carbon_footprint = 0

for r in results:
    boxes = r.boxes
    for box in boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100

        # Class Name
        cls = int(box.cls[0])
        if cls < len(classNames):  # Avoid out-of-range errors
            class_name = classNames[cls]
            item_footprint = carbon_footprint.get(class_name, 0)
            total_carbon_footprint += item_footprint

            # Display class name and confidence
            cvzone.putTextRect(
                image,
                f'{class_name} {conf} ({item_footprint} kg CO2e)',
                (max(0, x1), max(35, y1)),
                scale=2,
                thickness=2
            )

# Display total carbon footprint on the image
cvzone.putTextRect(
    image,
    f'Footprint: {total_carbon_footprint:.2f} kg CO2e',
    (20, 400),
    scale=2,
    thickness=2,
    colorR=(0, 255, 0)
)

# Resize the output image
# output_size = (800, 600)  # Desired output size (width, height)
# resized_image = cv2.resize(image, output_size)

# Save the resized output image
cv2.imwrite(output_path, image)

# Display the resized output image
cv2.imshow("Output Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
