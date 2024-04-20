import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import os
#from sort import Sort  # SORT algorithm for object tracking

# Give the path of the best.pt (best weights)
model_dir = "model"
model_file = "best.pt"
model_path = os.path.join(model_dir, model_file)

# Setting page layout
st.set_page_config(
    page_title="PPE(Private Protective Equipment)",  # Setting page title
    layout="wide",  # Setting layout to wide
    initial_sidebar_state="expanded",  # Expanding sidebar by default
)

try:
    model = YOLO(model_path)
    names = model.names
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Create a SORT tracker
tracker = Sort()

# Create a dictionary to store the count of detected objects
object_count = {}

# Creating sidebar
with st.sidebar:
    st.header("PPE detection system ")
    source = st.file_uploader(
        "Upload an image or video...", type=("jpg", "jpeg", "png", "bmp", "webp", "mp4")
    )
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

st.title("PPE(Private Protective Equipment) Detection")
st.caption("Upload a photo or video by selecting Browse files")
st.caption("Then click the Detect Objects button and check the result.")
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source is not None:
        file_extension = source.name.split(".")[-1]
        if file_extension in ["jpg", "jpeg", "png"]:
            uploaded_image = PIL.Image.open(source)
            image_width, image_height = uploaded_image.size
            st.image(
                uploaded_image,
                caption="Uploaded Image",
                width=image_width,
            )
            if st.sidebar.button("Detect Objects"):
                res = model.predict(
                    uploaded_image,
                    conf=confidence,
                    line_width=2,
                    show_labels=False,
                    show_conf=False,
                )
                boxes = res[0].boxes
                res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]

                # Update object count
                for box in boxes:
                    label = int(box.cls)
                    if label not in object_count:
                        object_count[label] = 1
                    else:
                        object_count[label] += 1

                with col2:
                    st.image(
                        res_plotted,
                        caption="Detected Image",
                        width=image_width,
                    )
                    try:
                        st.write(f"Number of detected: {len(boxes)}")
                        with st.expander("Classes Detected"):
                            for c in object_count.keys():
                                st.write(f"Class {c}: {object_count[c]}")
                    except Exception as ex:
                        st.write("No image is uploaded yet!")

        elif file_extension == "mp4":
            video_bytes = source.read()
            st.video(video_bytes)
            with open("input_video.mp4", "wb") as f:
                f.write(video_bytes)
            if st.sidebar.button("Detect Objects"):
                with col2:
                    vid_cap = cv2.VideoCapture("input_video.mp4")
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            image = cv2.resize(image, (720, int(720 * (9 / 16))))
                            res = model.predict(image, conf=confidence)
                            result_tensor = res[0].boxes
                            res_plotted = res[0].plot()
                            st_frame.image(
                                res_plotted,
                                caption="Detected Video",
                                channels="BGR",
                                use_column_width=True,
                            )

                            # Update object count
                            tracked_objects = tracker.update(result_tensor)
                            object_count.clear()
                            for obj in tracked_objects:
                                label = int(obj[4])
                                if label not in object_count:
                                    object_count[label] = 1
                                else:
                                    object_count[label] += 1

                            try:
                                st.write(f"Number of detected: {len(tracked_objects)}")
                                with st.expander("Classes Detected"):
                                    for c in object_count.keys():
                                        st.write(f"Class {c}: {object_count[c]}")
                            except Exception as ex:
                                st.write("No video is uploaded yet!")
                        else:
                            vid_cap.release()
                            break