import sys
import vpi
import numpy as np
from argparse import ArgumentParser
import cv2

from jetcam.csi_camera import CSICamera
from time import sleep
import datetime

if __name__=="__main__":
    # record images at maximum resolution
    fps = 21

    cam = CSICamera(width=4032, height=3040, capture_width=4032, capture_height=3040, capture_fps=fps)

    #cam.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    directory = "/home/isem/Desktop/billard-camera-module/config/calibration_images/"

    # take 15 images with 4s between each and save them with their current timestamp as .png
    i = 0
    j = 0
    s = 4
    while i < 15:
        frame = cam.read()
        j += 1
        #print(i, j)
        if j % fps == 0:
            print(s, "s left till pic", i+1)
            s -= 1

        if j == 8*fps:
            j = 0
            i += 1
            s = 8
            filename = "{:%Y-%m-%d_%H:%M:%S}.png".format(datetime.datetime.now())
            cv2.imwrite(directory + filename, frame)
            print("Image saved as", directory + filename)