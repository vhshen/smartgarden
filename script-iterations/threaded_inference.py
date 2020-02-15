#!/usr/bin/python

import os
import threading
import time
import sys
import cv2
from PIL import Image
from threading import Thread
from queue import LifoQueue
import re
import argparse

import jetson.inference
import jetson.utils


model = "--model=models/resnet18.onnx"
inp = "--input_blob=input_0"
outp = "--output_blob=output_0"
labels = "--labels=data/aminals/labels.txt"

# parse the command line
parser = argparse.ArgumentParser(description="Classify an image using an image recognition DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())
parser.add_argument("--file_in", type=str, help="filename of the input image to process")
opt = parser.parse_known_args()[0]

class VideoStream:
    # initialize the file video stream
    def __init__(self, queueSize=128):
        global capfps
        global capwid 
        global caphei
        cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
        # cap = cv2.VideoCapture(0)
        capfps = cap.get(cv2.CAP_PROP_FPS)
        capwid = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        caphei = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stream = cap
        self.stopped = False                                                                                                                
        # initialize the queue 
        self.Q = LifoQueue(maxsize=queueSize)                       

    # thread to read frames from stream
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):                    
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                
                # stop video if end of video file
                if not grabbed:
                    self.stop()
                    return
                                                                    
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()                                                                
    
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0                                                                                                           
    def clearQ(self):
        # empty the queue so it doesn't hit max size
        with self.Q.mutex:
            self.Q.queue.clear()
        return self.Q.empty()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

# load the recognition network
params = [model, inp, outp, labels]
net = jetson.inference.imageNet("resnet18", params)

# initialize pi camera video stream
vs = VideoStream().start()
time.sleep(2.0)

#initialize display
font = jetson.utils.cudaFont()

print("[INFO] looping over frames...")
while vs.more():
    img = vs.read()
    # vs.clearQ()
    # load an image (into shared CPU/GPU memory)
    # img, width, height = jetson.utils.loadImageRGBA(frame)

    width=1280
    height=720

    # classify the image
    class_idx, confidence = net.Classify(img, width, height)

    # find the object description
    class_desc = net.GetClassDesc(class_idx)

    # print out the result
    print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence\n".format(class_desc, class_idx, confidence * 100))

    # print out timing info
    # net.PrintProfilerTimes()

    font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence*100, class_desc), 5, 5, font.White, font.Gray40)
    cv2.imshow("Frame", img)
    key=cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
