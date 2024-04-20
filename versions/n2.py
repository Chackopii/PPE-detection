import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import os

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        names = model.names
        return model, names
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None, None

# Function to detect objects in an image
def detect_objects_in_image(model, uploaded_image, confidence, image_width):
    # Predict objects in the image
    res = model.predict(uploaded_image, conf=confidence, line_width=2, show_labels=False, show_conf=False)
    boxes = res[0].boxes
    res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
    # Resize the resulting image
    resized_image = PIL.Image.fromarray(res_plotted).resize((image_width, int(image_width * (9 / 16))))
    return resized_image, boxes

# Function to count the occurrences of each detected class
def count_classes(boxes, names):
    class_counts = {}
    for c in boxes.cls:
        class_name = names[int(c)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

# Function to detect objects in a video
def detect_objects_in_video(model, video_bytes, confidence):
    vid_cap = cv2.VideoCapture(video_bytes)
    frames = []
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (720, int(720 * (9 / 16))))
            res = model.predict(image, conf=confidence)
            res_plotted = res[0].plot()
            frames.append(res_plotted)
        else:
            vid_cap.release()
            break
    return frames

# Main function to run the Streamlit app
def main():
    # Path to the YOLO model
    model_dir = "model"
    model_file = "best.pt"
    model_path = os.path.join(model_dir, model_file)

    # Configure Streamlit page layout
    st.set_page_config(
        page_title="PPE(Private Protective Equipment)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load the YOLO model
    model, names = load_model(model_path)

    # If model failed to load, exit the app
    if model is None:
        return

    # Sidebar for uploading image or video and setting confidence
    with st.sidebar:
        st.header("Upload The Image")
        source = st.file_uploader(
            "Upload an image or video...", type=("jpg", "jpeg", "png", 'bmp', 'webp', 'mp4'))

        confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

    # Main title and instructions
    st.title("PPE(Private Protective Equipment) Detection")
    st.caption('Upload a photo or video by selecting :blue[Browse files]')
    st.caption('Then click the :blue[Detect Objects] button and check the result.')

    # Split page into two columns
    col1, col2 = st.columns(2)

    with col1:
        if source is not None:
            file_extension = source.name.split(".")[-1]
            if file_extension in ["jpg", "jpeg", "png"]:
                uploaded_image = PIL.Image.open(source)
                image_width, _ = uploaded_image.size
                st.image(uploaded_image, caption="Uploaded Image", width=image_width)

                if st.sidebar.button('Detect Objects'):
                    resized_image, boxes = detect_objects_in_image(model, uploaded_image, confidence, image_width)
                    class_counts = count_classes(boxes, names)
                    with col2:
                        st.image(resized_image, caption='Detected Image', width=image_width)
                        st.write(f'Number of detected objects: {len(boxes)}')
                        st.write('Detected Classes and Counts:')
                        for class_name, count in class_counts.items():
                            st.write(f'{class_name}: {count}')
            elif file_extension == "mp4":
                video_bytes = source.read()
                st.video(video_bytes)
                if st.sidebar.button("Detect Objects"):
                    frames = detect_objects_in_video(model, video_bytes, confidence)
                    with col2:
                        for frame in frames:
                            st.image(frame, caption="Detected Video", channels="BGR", use_column_width=True)

if _name_ == "_main_":
    main()