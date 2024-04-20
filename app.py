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

    col1, col2 = st.columns(2)

    with col1:
        if source is not None:
            file_extension = source.name.split(".")[-1]
            
            if file_extension in ["jpg", "jpeg", "png"]:
                uploaded_image = PIL.Image.open(source)
                image_width, _ = uploaded_image.size
                st.image(uploaded_image, 
                         caption="Uploaded Image", 
                         width=image_width)
                if st.sidebar.button('Detect Objects'):
                    res_plotted, boxes = detect_objects_in_image(model, uploaded_image, confidence, image_width)
                    with col2:
                        st.image(res_plotted, 
                                 caption='Detected Image', 
                                 width=image_width)
                        try:
                            st.write(f'Number of detected: {len(boxes)}')
                            with st.expander("class detected:"):
                                for c in boxes.cls:
                                    st.write(names[int(c)])
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
                       
                        frames = detect_objects_in_video(model, vid_cap, confidence)
                        for frame in frames:
                            st.image(frame, caption="Detected Video", channels="BGR", use_column_width=True)


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
    return res_plotted, boxes

def detect_objects_in_video(model,  vid_cap, confidence):
    # vid_cap = cv2.VideoCapture(video_bytes)
    frames = st.empty()
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (720, int(720 * (9 / 16))))
            res = model.predict(image, conf=confidence)
            result_tensor = res[0].boxes
            res_plotted = res[0].plot()
            frames.image(res_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
        else:
            vid_cap.release()
            break

# def main():
#     model_dir = "model"
#     model_file = "best.pt"
#     model_path = os.path.join(model_dir, model_file)

#     st.set_page_config(
#         page_title="PPE(Private Protective Equipment)",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )

#     model, names = load_model(model_path)

#     if model is None:
#         return

#     with st.sidebar:
#         st.header("Upload The Image")
#         source = st.file_uploader(
#             "Upload an image or video...", type=("jpg", "jpeg", "png", 'bmp', 'webp', 'mp4'))

#         confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

#     st.title("PPE(Private Protective Equipment) Detection")
#     st.caption('Upload a photo or video by selecting :blue[Browse files]')
#     st.caption('Then click the :blue[Detect Objects] button and check the result.')

#     col1, col2 = st.columns(2)

    # with col1:
    #     if source is not None:
    #         file_extension = source.name.split(".")[-1]
            
    #         if file_extension in ["jpg", "jpeg", "png"]:
    #             uploaded_image = PIL.Image.open(source)
    #             image_width, _ = uploaded_image.size
    #             st.image(uploaded_image, 
    #                      caption="Uploaded Image", 
    #                      width=image_width)
    #             if st.sidebar.button('Detect Objects'):
    #                 res_plotted, boxes = detect_objects_in_image(model, uploaded_image, confidence, image_width)
    #                 with col2:
    #                     st.image(res_plotted, 
    #                              caption='Detected Image', 
    #                              width=image_width)
    #                     try:
    #                         st.write(f'Number of detected: {len(boxes)}')
    #                         with st.expander("class detected:"):
    #                             for c in boxes.cls:
    #                                 st.write(names[int(c)])
    #                     except Exception as ex:
    #                           st.write("No image is uploaded yet!")
                            

    #         elif file_extension == "mp4":
    #             video_bytes = source.read()
    #             st.video(video_bytes)
    #             with open("input_video.mp4", "wb") as f:
    #                 f.write(video_bytes)
    #             if st.sidebar.button("Detect Objects"):
    #                 with col2:
    #                     frames = detect_objects_in_video(model, f, confidence)
    #                     for frame in frames:
    #                         st.image(frame, caption="Detected Video", channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
