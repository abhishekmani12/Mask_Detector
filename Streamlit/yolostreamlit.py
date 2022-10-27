import torch
import yaml
import pandas as pd
import os
from io import StringIO
from pathlib import Path
import streamlit as st
from PIL import Image
import ffmpeg
import cv2
import numpy as np
from mss import mss


st.title("Mask Detector")
st.sidebar.title('Configure')


savePath="C:/Users/Abhis/Desktop/YOLO/Code/yolov5/runs/detect/exp"
#inputPath="C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages"
label=''
spinnerlabel=''

source = ("Image [Bbox Output]", "Video","Webcam", "Image [CSV Output]")
source_index = st.sidebar.selectbox("Input", range(len(source)), format_func=lambda x: source[x])
st.sidebar.write("####")

confLevel=st.sidebar.slider("Confidence Slider - Optimized value is 0.4", 0.0, 1.0, 0.4)
st.sidebar.write("####")

if source_index == 0 :

        label='Submit'
        st.write('See detections in an Image')
        st.sidebar.write("#####")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            is_valid = True

            with st.spinner(text='Loading'):
                
                picture = Image.open(uploaded_file)
                picture = picture.save(f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}')
                sourceInterp = f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}'
                

        else:
            is_valid = False

elif source_index == 1:

        label='Submit'
        st.write('See detections in a Video')
        st.sidebar.write("#####")
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])

        if uploaded_file is not None:
            is_valid = True

            with st.spinner(text='Loading'):
                
                with open(os.path.join("C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                sourceInterp = f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}'
                
        else:
            is_valid = False

elif source_index == 2:

        st.write('See live detections using your webcam')
        st.sidebar.write("#####")
        is_valid=True
        sourceInterp='0'
        label='Ready' 

elif source_index == 3:

        st.write("See detections listed in an interactive table")
        st.sidebar.write("#####")
        label='Submit'

        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            is_valid = True

            with st.spinner(text='Loading'):
                picture = Image.open(uploaded_file)
                picture = picture.save(f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}')
                sourceInterp = f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}'
                

        else:
            is_valid = False        
                 





if is_valid:
        print('valid')
        st.sidebar.text("  ")
        if st.sidebar.button(label):

            delpath=savePath + '/'

            for file_name in os.listdir(delpath):
            # construct full file path
                file = delpath + file_name
                if os.path.isfile(file):
                    os.remove(file)

            if source_index == 2:
                spinnerlabel='Live Video Stream Enabled \n\n Check for a new popup window to see the live video \n\n Press "Q" to stop the stream' 
            else:
                spinnerlabel='Processing'       

            with st.spinner(text=spinnerlabel):
                
                if source_index == 3:
                    model=torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Abhis/Desktop/YOLO/weights/best1.pt')
                    model.conf=confLevel
                else:    
                    os.system(f"python C:/Users/Abhis/Desktop/YOLO/Code/yolov5/detect.py --weights C:/Users/Abhis/Desktop/YOLO/weights/best1.pt --source {sourceInterp} --conf {confLevel}")

                if source_index == 0: #IMAGE
                    with st.spinner(text='Processing'):

                        for img in os.listdir(savePath):

                            fullpath=f'{savePath}/' + img
                            IMAGE=Image.open(fullpath)
                            st.image(IMAGE)
                            
                        st.sidebar.write("######")
                        st.success("Process Completed",  icon="✅")

                elif source_index == 1: #VIDEO
                    with st.spinner(text='Converting codec'):
                        
                        for vid in os.listdir(savePath):

                            fullpath=f'{savePath}/' + vid

                            (
                                ffmpeg
                                .input(fullpath)
                                .filter('fps', fps=25, round='up')
                                .output('C:/Users/Abhis/Desktop/YOLO/Code/yolov5/runs/detect/exp/nicecodec.mp4', vcodec='libx264')
                                .run()
                            )                           

                            fullpath =f'{savePath}/' + 'nicecodec.mp4'

                            video_file = open(fullpath, 'rb')
                            video_bytes = video_file.read()
                            
                            st.video(video_bytes)
                            
                            st.sidebar.write("######")
                            st.success("Process Completed",  icon="✅")

                elif source_index == 2: #WEBCAM

                        st.warning('Webcam has stopped')
                        st.write("Playback of video feed from webcam")
                        st.sidebar.write("######")

                    

                        for vid in os.listdir(savePath):

                            fullpath=f'{savePath}/' + vid

                            (
                                ffmpeg
                                .input(fullpath)
                                .filter('fps', fps=25, round='up')
                                .output('C:/Users/Abhis/Desktop/YOLO/Code/yolov5/runs/detect/exp/nicecodec.mp4', vcodec='libx264')
                                .run()
                            )                           

                            fullpath =f'{savePath}/' + 'nicecodec.mp4'

                            video_file = open(fullpath, 'rb')
                            video_bytes = video_file.read()
                            
                            st.video(video_bytes)
                            
                            st.sidebar.write("######")
                            st.success("Process Completed",  icon="✅")


                elif source_index == 3:
                    with st.spinner("Inferencing"):


                        res=model(sourceInterp)
                        result=res.pandas().xyxy[0]
                        df=pd.DataFrame(result)
                    st.dataframe(df)
                    st.sidebar.write("######")
                    st.success("Process Completed",  icon="✅")