import numpy as np
import av
import mtcnn
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

#cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Detectionmodel=mtcnn.MTCNN()
Basemodel=torch.load('model.pth', map_location=torch.device('cpu'))
Basemodel.eval()

#image tsfm for basemodel

classes = ['With_Mask', 'Without_Mask']

tsfm = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean across each RGB channel
                         [0.229, 0.224, 0.225])
])

#base model pred function:

def pred(img, tsfm):
    tensor_img = tsfm(img)

    tensor_img = torch.unsqueeze(tensor_img,0) #only for single image pred rather than batch pred
    

    imginput = Variable(tensor_img)

    prediction = Basemodel(imginput)
    index = prediction.data.numpy().argmax()
    #print(index)
    output = classes[index]

    return output

#detection method:

def detect(frame):
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces=Detectionmodel.detect_faces(img) #face detection
    
    for result in faces:
        x,y,w,h=result['box']
    
        roi_color = frame[y:y+h, x:x+w]
        roi_resize = cv2.resize(roi_color, (224, 224))
        
        output=pred(roi_resize, tsfm) # base model detection
        #print(output)
        
        label = "Masked" if output=='With_Mask' else "Not Masked!"
        
        colour = (0, 255, 0) if label == "Masked" else (0, 0, 255)
        
        cv2.putText(frame, label, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        
        cv2.rectangle(frame,(x,y),(x+w, y+h), colour,4)
    
    return frame

RTC_CONFIGURATION = RTCConfiguration
(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        fimg = detect(frame)

        return av.VideoFrame.from_ndarray(fimg, format="rgb24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,

)
