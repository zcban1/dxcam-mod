import cupy as cp
from .base import Processor
import ctypes

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
            
        buffer_ptr = ctypes.addressof(rect.pBits.contents) + offset
        buffer = cp.frombuffer((ctypes.c_char * size).from_address(buffer_ptr), dtype=cp.uint8)

        # Pointer arithmetic
        pitch = pitch // 4
        if rotation_angle in (0, 180):
            image = cp.asarray(buffer, dtype=cp.uint8).reshape((height, pitch, 4))
        elif rotation_angle in (90, 270):
            image = cp.asarray(buffer, dtype=cp.uint8).reshape((width, pitch, 4))

        # if not self.color_mode is None:
        image = image[:, :, :3]

        if rotation_angle == 90:
            image = cp.rot90(image, axes=(1, 0))
        elif rotation_angle == 180:
            image = cp.rot90(image, k=2, axes=(0, 1))
        elif rotation_angle == 270:
            image = cp.rot90(image, axes=(0, 1))

        if rotation_angle in (0, 180) and pitch != width:
            image = image[:, :width, :]
        elif rotation_angle in (90, 270) and pitch != height:
            image = image[:height, :, :]

        if region[3] - region[1] != image.shape[0]:
            image = image[region[1]: region[3], :, :]
        if region[2] - region[0] != image.shape[1]:
            image = image[:, region[0]: region[2], :]

        return image
