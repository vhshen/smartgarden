#!/usr/bin/python

import jetson.inference
import jetson.utils

import argparse
import sys

from playsound import playsound
import subprocess

# function that triggers when a specific class is detected
def repel(class_id):
    if class_id == "Bird":
        # birds are repelled by shiny / bright lights
        

    elif class_id == "Deer":
        # deer are repelled by the sound of dogs barking
        playsound('dogbark.mp3')

    elif class_id == "Dog":
        # dogs are repelled by water getting sprayed at them

    elif class_id == "Squirrel":
        # squirrels are repelled by motion systems
        subprocess.call("./fanon.sh")

    elif class_id == "Rabbit":
        # lights are repelled by the sound of dogs barking and motion
        playsound('dogbark.mp3')
        subprocess.call("./fanon.sh")

# cancels all the "effects" of the repel class
def reset():
    subprocess.call("./fanoff.sh")

model = "--model=models/resnet18.onnx"
inp = "--input_blob=input_0"
outp = "--output_blob=output_0"
labels = "--labels=data/aminals/labels.txt"

params = [model, inp, outp, labels]
# load the recognition network
net = jetson.inference.imageNet("resnet18", params)

# create the camera and display
font = jetson.utils.cudaFont()
camera = jetson.utils.gstCamera(1280, 720, "0")
display = jetson.utils.glDisplay()


counter = 10000
class_desc = "x"
# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

        counter += 1

        if counter > 7200:
            # only run detection if it's been more than 1 minute since last detection
	    
            reset()

            # classify the image
	    class_idx, confidence = net.Classify(img, width, height)
            # find the object description
	    class_desc = net.GetClassDesc(class_idx)

            if confidence*100 > 20:
                # detection only counts if the network is more than 20% confident
                # the threshold is so low because the model is not very accurate :(
                counter = 0
                repel(class_desc)

	# overlay the result on the image	
	font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        
