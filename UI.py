import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
import cvzone
import math

st.title("Carbon Detection in Clothing")

st.markdown("""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 350px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 350px;
    margin-left: -350px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Options")
st.sidebar.subheader("Input Parameters")
model = YOLO("Model and Results/best1.pt")
classNames = ['dress', 'jacket', 'pants', 'polo', 'shirt', 'shoes', 'shorts', 'tshirt']
carbon_footprint = {
   "dress": 15,       # Cotton dress
    "jacket": 30,      # Polyester jacket (synthetic materials have higher footprints)
    "pants": 33,       # Denim pants (denim production is water and energy-intensive)
    "polo": 12,        # Cotton polo shirt
    "shirt": 10,       # Casual shirt
    "shoes": 30,       # Leather shoes (leather has high carbon and water footprints)
    "shorts": 8,       # Cotton shorts
    "tshirt": 8        # Cotton T-shirt
}

upload_or_capture = st.sidebar.radio("Upload or Capture Image", ("Upload Image", "Capture Using Webcam"))

if upload_or_capture == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name
    else:
        image_path = None
elif upload_or_capture == "Capture Using Webcam":
    st.sidebar.text("Use Streamlit's `st.camera_input` for camera input.")
    captured_image = st.camera_input("Capture an Image")
    if captured_image:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(captured_image.read())
        image_path = temp_file.name
    else:
        image_path = None

if image_path:
    image = cv2.imread(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    results = model(image)
    total_carbon_footprint = 0
    detected_items = []  

    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            if cls < len(classNames): 
                class_name = classNames[cls]
                item_footprint = carbon_footprint.get(class_name, 0)
                total_carbon_footprint += item_footprint

                detected_items.append((class_name, item_footprint))

                cvzone.putTextRect(
                    image,
                    f'{class_name}',
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=2
                )

    st.image(image, caption="Processed Image (Class Names Displayed)", use_column_width=True)

    st.write("### Detected Items and Carbon Content")

    col1, col2, col3 = st.columns(3)

    col1.write("**Clothes Name**")
    col2.write("**Carbon Footprint**")
    col3.write("**Cumulative Carbon Footprint**")

    cumulative_footprint = 0
    for class_name, item_footprint in detected_items:
        cumulative_footprint += item_footprint
        col1.write(class_name)
        col2.write(f"{item_footprint} kg CO2e")
        col3.write(f"{cumulative_footprint:.2f} kg CO2e")

else:
    st.write("Please upload an image or capture one using your camera.")
