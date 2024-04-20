import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import os

def main():
    model_dir = "model"
    model_file = "best.pt"
    model_path = os.path.join(model_dir, model_file)

    st.set_page_config(
        page_title="PPE(Private Protective Equipment)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    model, names = load_model(model_path)

    if model is None:
        return

    with st.sidebar:
        st.header("Upload The Image")
        source = st.file_uploader(
            "Upload an image or video...", type=("jpg", "jpeg", "png", 'bmp', 'webp', 'mp4'))

        confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

    st.title("PPE(Private Protective Equipment) Detection")
    st.caption('Upload a photo or video by selecting :blue[Browse files]')
    st.caption('Then click the :blue[Detect Objects] button and check the result.')

    # col1 = st.columns(1)

    # with col1:
    # imageLocation = st.empty()
    if source is not None:
        file_extension = source.name.split(".")[-1]
        
        if file_extension in ["jpg", "jpeg", "png",]:
            uploaded_image = PIL.Image.open(source)
            image_width,image_height = uploaded_image.size
            # imageLocation.image(uploaded_image)
            # uploaded_image=uploaded_image.resize((640, int(480 *(image_width/image_height))))
            uploaded_image=uploaded_image.resize((480, int(640 * (9 / 16))))
            st.image(uploaded_image, 
                        caption="Uploaded Image", 
                        # width=image_width,
                        )
            
            if st.sidebar.button('Detect Objects'):
                resized_image, boxes = detect_objects_in_image(model, uploaded_image, confidence, image_width)
                class_counts = count_classes(boxes, names)
                # res_plotted, boxes = detect_objects_in_image(model, uploaded_image, confidence, image_width)
                # with col2:
                if resized_image is not None:
                    # imageLocation.image(resized_image)
                    st.subheader("Detected images")
                    st.image(resized_image, 
                             caption='Detected Image', 
                            #  width=image_width,
                            )
                    # st.image(res_plotted, 
                    #          caption='Detected Image', 
                    #          width=image_width)
                    try:
                        st.write(f'Number of detected: {len(boxes)}')
                        with st.expander('Detected Classes and Counts: '):
                            for class_name, count in class_counts.items():
                                st.write(f'{class_name}: {count}')
                            # for c in boxes.cls:
                            #     st.write(names[int(c)])
                    except Exception as ex:
                            st.write("No image is uploaded yet!")
                            

        elif file_extension == "mp4":
            video_bytes = source.read()
            st.video(video_bytes,)
            with open("input_video.mp4", "wb") as f:
                f.write(video_bytes)
            if st.sidebar.button("Detect Objects"):
                # with col2:
                vid_cap = cv2.VideoCapture("input_video.mp4")
                
                frames = detect_objects_in_video(model, vid_cap, confidence)
                for frame in frames:
                    st.image(frame, caption="Detected Video", channels="BGR", use_column_width=True,)


def load_model(model_path):
    try:
        model = YOLO(model_path)
        names = model.names
        return model, names
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
        return None, None

def detect_objects_in_image(model, uploaded_image, confidence, image_width):
    res = model.predict(uploaded_image, conf=confidence, line_width=2, show_labels=False, show_conf=False)
    boxes = res[0].boxes
    res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
    # Resize the resulting image
    resized_image = PIL.Image.fromarray(res_plotted).resize((480, int(640 * (9 / 16))))
    return resized_image, boxes
    # return res_plotted, boxes

# Function to count the occurrences of each detected class
def count_classes(boxes, names):
    class_counts = {}
    for c in boxes.cls:
        class_name = names[int(c)]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

def detect_objects_in_video(model,  vid_cap, confidence):
    # vid_cap = cv2.VideoCapture(video_bytes)
    frames = st.empty()
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            #image = cv2.resize(image, (720, int(720 * (9 / 16))))
            res = model.track(image, conf=confidence,persist=True,tracker="bytetrack.yaml")
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            frames.image(res_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

if __name__ == "__main__":
    main()
