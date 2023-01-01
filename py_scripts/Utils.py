import glob
import os
import sys
import csv

import random
from tkinter import W
from turtle import pos
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import cv2
from PIL import Image

import torch

from env_carla import Environment


# returns all absolute paths in #paths
def get_image_paths(path, filter="*"):
    path_list = []

    for root, dirs, files in os.walk(os.path.abspath(path)):
        for file in files:
            if filter == "*":
                path_list.append(os.path.join(root, file))
            else:
                path = os.path.join(root, file)
                endpoint = path.split(".")[-1]
                if endpoint == filter:
                    path_list.append(path)
    return path_list

# takes in a figure and converts it to an image (int)
def plotToImage(figure):
    canvas = figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    return image