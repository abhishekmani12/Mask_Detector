import cv2
import streamlit as st
import mtcnn
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Detectionmodel=mtcnn.MTCNN()
Basemodel=torch.load(r'C:\Users\Abhis\Desktop\detector\MaskDetStuff\model.pth', map_location=torch.device('cpu'))
Basemodel.eval()


classes = ['With_Mask', 'Without_Mask']

tsfm = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean across each RGB channel
                         [0.229, 0.224, 0.225])
])

def pred(img, tsfm):
    tensor_img = tsfm(img)

    tensor_img = torch.unsqueeze(tensor_img,0) #only for single image pred rather than batch pred
    

    imginput = Variable(tensor_img)

    prediction = Basemodel(imginput)
    index = prediction.data.numpy().argmax()
    #print(index)
    
    return classes[index]




st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cap= cv2.VideoCapture(0)

while run:
    _, frame=cap.read()
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces=Detectionmodel.detect_faces(img)

    #for mtcnn
    for result in faces:
        x,y,w,h=result['box']
    
        roi_color = frame[y:y+h, x:x+w]
        roi_resize = cv2.resize(roi_color, (224, 224))
        
        output=pred(roi_resize, tsfm)
        #print(output)
        
        label = "Masked" if output=='With_Mask' else "Not Masked!"
        
        colour = (0, 255, 0) if label == "Masked" else (0, 0, 255)
        
        cv2.putText(frame, label, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        
        cv2.rectangle(frame,(x,y),(x+w, y+h), colour,4)

        frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
