#!/usr/bin/python

import jetson.inference
import jetson.utils

import argparse

model = "--model=jetson-inference/models/resnet18.onnx"
inp = "--input_blob=input_0"
outp = "--output_blob=output_0"
labels = "--labels=data/aminals/labels.txt"

# parse the command line
parser = argparse.ArgumentParser(description="Classify an image using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("file_in", type=str, default="rt", help="filename of the input image or video to process")

opt = parser.parse_known_args()[0]

#if opt.file_in == "rt":


# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

params = [model, inp, outp, labels, opt.file_in]
# load the recognition network
net = jetson.inference.imageNet("resnet18", params)

# classify the image
class_idx, confidence = net.Classify(img, width, height)

# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence\n".format(class_desc, class_idx, confidence * 100))

# print out timing info
net.PrintProfilerTimes()


