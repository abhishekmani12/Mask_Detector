import torch
from torch.autograd import Variable
from torchvision.transforms import transforms
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

#pip install mtcnn
from mtcnn.mtcnn import MTCNN
import os

classes = ['With_Mask', 'Without_Mask']

tsfm = transforms.Compose([

	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406],  # mean across each RGB channel
						 [0.229, 0.224, 0.225])
])


def pred(img, tsfm):
	tensor_img = tsfm(img)

	#tensor_img = tensor_img.unsqueeze(0)

	imginput = Variable(tensor_img)

	prediction = Basemodel(imginput)
	index = prediction.data.numpy().argmax()
	print(index)
	output = classes[index]

	return output

def predict(frame, Detectionmodel, Basemodel):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	#Detectionmodel.setInput(blob)
	#detections = Detectionmodel.forward()

	detections=Detectionmodel.detect_faces(blob)
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			'''face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)'''

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop

		preds=pred(faces, tsfm)


	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
Detectionmodel=MTCNN()

# load the face mask detector model from disk
Basemodel = torch.load('model.pth',map_location=torch.device('cpu'))
Basemodel.eval()

# initialize the video stream
print(" Starting video stream")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = predict(frame, Detectionmodel, Basemodel)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(With_Mask, without_Mask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Masked" if without_Mask>With_Mask else "Not Masked!"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		#label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("f"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
