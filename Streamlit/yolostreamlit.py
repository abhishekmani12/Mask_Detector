import torch
import yaml
import pandas
import os
from io import StringIO
from pathlib import Path
import streamlit as st
import time
import sys
import argparse
from PIL import Image



st.title('Mask Detection')

savePath="C:/Users/Abhis/Desktop/YOLO/Code/yolov5/runs/detect/exp"
#inputPath="C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages"

source = ("Image", "Video")
source_index = st.sidebar.selectbox("Input", range(len(source)), format_func=lambda x: source[x])
confLevel=st.sidebar.slider("Confidence Level Range - Optimized value is 0.4", 0.0, 1.0, 0.4)

if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            is_valid = True

            with st.spinner(text=''):
                #st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}')
                sourceInterp = f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}'
                

        else:
            is_valid = False
else:
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])

        if uploaded_file is not None:
            is_valid = True

            with st.spinner(text=''):
                #st.sidebar.video(uploaded_file)

                with open(os.path.join("C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                sourceInterp = f'C:/Users/Abhis/Desktop/YOLO/Code/yolov5/InputImages/{uploaded_file.name}'
                
        else:
            is_valid = False




if is_valid:
        print('valid')
        if st.sidebar.button('Submit'):
            with st.spinner(text='Processing'):

                delpath=savePath + '/'

                for file_name in os.listdir(delpath):
                # construct full file path
                    file = delpath + file_name
                    if os.path.isfile(file):
                        os.remove(file)

                os.system(f"python C:/Users/Abhis/Desktop/YOLO/Code/yolov5/detect.py --weights C:/Users/Abhis/Desktop/YOLO/weights/best1.pt --source {sourceInterp} --conf {confLevel}")

                if source_index == 0:
                    with st.spinner(text='Processing'):
                    
                        for img in os.listdir(savePath):

                            fullpath=f'{savePath}/' + img
                            IMAGE=Image.open(fullpath)
                            st.image(IMAGE)
                            

                        st.success("Process Completed")
                else:
                    with st.spinner(text='Processing'):
                        
                        for vid in os.listdir(savePath):

                            fullpath=f'{savePath}\' + vid

                            
                            video_file = open(fullpath, 'rb')
                            video_bytes = video_file.read()
                            
                            st.video(video_bytes)
                            
                            st.success("Process Completed")


