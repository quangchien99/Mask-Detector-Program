import sys
from tkinter import Button, Entry, Toplevel, Label, Tk

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def detect_and_predict_mask(frame, faceNet, maskNet):
    # get the dimensions of the frame =>construct a point
    (h, w) = frame.shape[:2]
    point = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # put  blob into the network and get the face detections
    faceNet.setInput(point)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # resize it to 224x224
            # extract the face ROI, convert it from BGR to RGB
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Detecting when at least 1 face exist
    if len(faces) > 0:
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="face detector")
ap.add_argument("-m", "--model", type=str,
                default="model.model",
                help="trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
protoPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(protoPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# get the video stream from webcam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
count = 0
# loop over the frames from webcam video
while True:
    # resize the frame to maximum is 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1200)

    # detect
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color to the video from webcam
        label = "Mask On" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask On" else (0, 0, 255)

        # get the probability displaying with label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        date = str(datetime.datetime.now())
        cv2.putText(frame, date, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        # display theq label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (5, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # display
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    saveImage = cv2.waitKey(1) & 0xFF
    # press 's' in 1 seconds to save capture image of video
    if saveImage == ord("s"):
        locationSave = 'imagesCaptured'
        out = cv2.imwrite(os.path.join(locationSave, "Image %d.jpg" % count), frame)
        print("Image %d.img" % count + " saved")
        count = count + 1

    # press q to quit
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
