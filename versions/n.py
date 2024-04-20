import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import os
#import tempfile

# Give the path of the best.pt (best weights)
model_dir="model"
model_file="best.pt"
model_path = os.path.join(model_dir, model_file)

# Setting page layout
st.set_page_config(
    page_title="PPE(Private Protective Equipment)",  # Setting page title
    #page_icon="NK logo.jpeg",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)

try:
    model = YOLO(model_path)
    names=model.names
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


# Creating sidebar
with st.sidebar:
    st.header("Upload The Image")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source = st.file_uploader(
        "Upload an image or video...", type=("jpg", "jpeg", "png", 'bmp', 'webp','mp4'))
    
    #st.sidebar("Upload the video")   #adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    #source_video=st.file_uploader("Upload a video...", type=("mp4"))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("PPE(Private Protective Equipment) Detection")
st.caption('Updload a photo or video by selecting :blue[Browse files]')
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
     #checking if the source is not empty
    if source is not None:

        #getting the file extentions from the uploaded file using name.split()
        file_extension = source.name.split(".")[-1]

        #checking if the file extention is image or not
        if file_extension in ["jpg", "jpeg", "png"]:

            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source)

            # Getting the image size
            image_width, image_height = uploaded_image.size

            # Adding the uploaded image to the page with a caption
            st.image(uploaded_image,
                    caption="Uploaded Image",
                    width=image_width
                    )
            #checking if the the button is clicked
            if st.sidebar.button('Detect Objects'):
                #prediction based on the image with conf=confidence from the slider
                res = model.predict(uploaded_image,
                                    conf=confidence,
                                    line_width=2, 
                                    show_labels=False, 
                                    show_conf=False
                                    )
                #extracting information about the bounding box from res 
                boxes = res[0].boxes
                #plotting the bounding box with confidence and labels
                res_plotted = res[0].plot(labels=True, line_width=1)[:, :, ::-1]
                with col2:
                    st.image(res_plotted,
                            caption='Detected Image',
                            width=image_width                 
                            )
                    try:
                        st.write(f'Number of detected: {len(boxes)}')
                        with st.expander("class detected:"):                
                            for c in boxes.cls:
                                st.write(names[int(c)])
                                #print(names[int(c)])
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
            
        elif file_extension == "mp4":

            video_bytes = source.read()
            st.video(video_bytes)
            # Save video locally
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
                            res = model.track(image, conf=confidence, persist=True,tracker="bytetrack.yaml")
                            result_tensor = res[0].boxes
                            res_plotted = res[0].plot()
                            st_frame.image(res_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
                        else:
                            vid_cap.release()
                            break
