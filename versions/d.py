
from ultralytics import YOLO
import streamlit as st
from PIL import Image, ImageDraw
import torch
import cv2
from io import BytesIO
def detect_objects_image(image):
    # Load YOLOv5 model
    model = YOLO('model/best.pt')
    # Perform object detection
    Results = model.predict(image,show=True,save=True)
    return Results


def detect_objects_video(video):
    # Load YOLOv8 model
    model = YOLO('model/best.pt')

    # Open video capture
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on each frame
        results = model(frame)

        # Draw bounding boxes on the frame
        frame = results.render()[0]

        # Write the frame into the file 'output.avi'
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    st.title("PPE Object Detection")

    # Upload image or video
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    #checking if the uploaded file is not empty
    if uploaded_file is not None:
        #getting the file extentions from the uploaded file using name.split() 
        file_extension = uploaded_file.name.split(".")[-1] 
        #checking if the file extention is image or not
        if file_extension in ["jpg", "jpeg", "png"]:
            # Display uploaded image
            image = Image.open(uploaded_file)
            #
            st.image(image, caption="Uploaded Image", use_column_width=True)
            submit=st.button('predict')
            if submit:
                # Perform object detection on the image
                results = detect_objects_image(image)
                # Display object detection results
                st.subheader("Object Detection Results:")
                #st.image(results)
                
                #for r in results:  
        elif file_extension == "mp4":
            # Display uploaded video
            video_bytes = uploaded_file.read()
            st.video(video_bytes)

            # Save video locally
            with open("input_video.mp4", "wb") as f:
                f.write(video_bytes)

            # Perform object detection on the video
            detect_objects_video("input_video.mp4")

            # Display a link to download the processed video
            st.markdown("[Download Processed Video](output.mp4)")


if __name__ == "__main__":
    main()