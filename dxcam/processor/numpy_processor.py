import numpy as np 
from .base import Processor
import ctypes
import cv2

class NumpyProcessor(Processor):
    def __init__(self, color_mode):
        self.cvtcolor = None
        self.color_mode = color_mode


    def process_cvtcolor(self, image):
        # Remove the alpha channel by selecting only the first 3 channels (BGR)
        image = image[:, :, :3]
        return image

    def process(self, rect, width, height, region, rotation_angle):
        pitch = int(rect.Pitch)

        if rotation_angle in (0, 180):
            offset = (region[1] if rotation_angle == 0 else height - region[3]) * pitch
            height = region[3] - region[1]
            size = pitch * height
        else:
            offset = (region[0] if rotation_angle == 270 else width - region[2]) * pitch
            width = region[2] - region[0]
            size = pitch * width

        buffer = (ctypes.c_char*size).from_address(ctypes.addressof(rect.pBits.contents)+offset)#Pointer arithmetic
        pitch = pitch // 4

        image = np.ndarray((height, pitch, 4), dtype=np.uint8, buffer=buffer)


        # Evitare ridimensionamenti non necessari
        if rotation_angle in (0, 180):
            if pitch != width:
                image = image[:, :width, :]
        else:
            if pitch != height:
                image = image[:height, :, :]

        # Applicare la rotazione
        if rotation_angle == 90:
            image = np.rot90(image, axes=(1, 0))
        elif rotation_angle == 180:
            image = np.rot90(image, k=2, axes=(0, 1))
        elif rotation_angle == 270:
            image = np.rot90(image, axes=(0, 1))

        # Tagliare i bordi dell'immagine
        if region[3] - region[1] != image.shape[0]:
            image = image[region[1] : region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0] : region[2], :]

        image = image[:, :, :3]
        #convert to hsv
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return image




