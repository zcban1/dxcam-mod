import time
import ctypes
from typing import Tuple
from threading import Thread, Event, Lock
import comtypes
import numpy as np
from dxcam.core import Device, Output, StageSurface, Duplicator
from dxcam.processor import Processor
from dxcam.util.timer import (
    create_high_resolution_timer,
    set_periodic_timer,
    wait_for_timer,
    cancel_timer,
    INFINITE,
    WAIT_FAILED,
)
import cupy as cp


class DXCamera:
    def __init__(
        self,
        output: Output,
        device: Device,
        region: Tuple[int, int, int, int],
        output_color: str = "RGB",
        max_buffer_len=300,
    ) -> None:
        self._output: Output = output
        self._device: Device = device
        self.width, self.height = self._output.resolution
        self._stagesurf: StageSurface = StageSurface(width=self.width, height=self.height,
            output=self._output, device=self._device
        )
        self._duplicator: Duplicator = Duplicator(
            output=self._output, device=self._device
        )
        self._processor: Processor = Processor(output_color=output_color)

        
        self.channel_size = len(output_color) #if output_color != "GRAY" else 1
        self.rotation_angle: int = self._output.rotation_angle

        self._region_set_by_user = region is not None
        self.region: Tuple[int, int, int, int] = region

        self.max_buffer_len = max_buffer_len
        self.is_capturing = False

        self.__thread = None
        self.__lock = Lock()
        self.__stop_capture = Event()

        self.__frame_available = Event()
        self.__frame_buffer: np.ndarray = None
        self.__head = 0
        self.__tail = 0
        self.__full = False

        self.__timer_handle = None

        self.__frame_count = 0
        self.__capture_start_time = 0

    def grab(self, region: Tuple[int, int, int, int] = None):
        frame = self._grab(region)
        if frame is not None:
            return frame

    def _grab(self, region: Tuple[int, int, int, int]):
        if self._duplicator.update_frame():
            if not self._duplicator.updated:
                return None

            self._device.im_context.CopyResource(self._stagesurf.texture, self._duplicator.texture)
            self._duplicator.release_frame()
            rect = self._stagesurf.map()
            frame = self._processor.process(rect, self.width, self.height, region, self.rotation_angle)
            self._stagesurf.unmap()
            return frame



    def start(
        self,
        region: Tuple[int, int, int, int] = None,
        target_fps: int = 0,
        video_mode=False,
    ):

        self.is_capturing = True
        frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
        self.__frame_buffer = cp.ndarray(
            (self.max_buffer_len, *frame_shape), dtype=np.uint8
        )
        self.__thread = Thread(
            target=self.__capture,
            name="DXCamera",
            args=(region, target_fps, video_mode),
        )
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        if self.is_capturing:
            self.__frame_available.set()
            self.__stop_capture.set()
            if self.__thread is not None:
                self.__thread.join(timeout=1)
        self.is_capturing = False
        self.__frame_buffer = None
        self.__frame_count = 0
        self.__frame_available.clear()
        self.__stop_capture.clear()

    def get_latest_frame(self):
        self.__frame_available.wait()
        #with self.__lock:
        if self.__frame_count > 0:
            ret = self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
            self.__frame_available.clear()
            self.__frame_count -= 1
            return cp.array(ret)
        return None

    def __capture(self, region: Tuple[int, int, int, int], target_fps: int = 0, video_mode=False):
        while not self.__stop_capture.is_set():
            try:
                frame = self._grab(region)
                if frame is None and video_mode:
                    frame = cp.array(self.__frame_buffer[(self.__head - 1) % self.max_buffer_len])

                #with self.__lock:
                self.__frame_buffer[self.__head] = frame
                self.__head = (self.__head + 1) % self.max_buffer_len
                self.__full = self.__head == self.__tail
                self.__frame_available.set()
                self.__frame_count += 1

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.__stop_capture.set()
                raise e

        fps = int(self.__frame_count / (time.perf_counter() - self.__capture_start_time))
        print(f"Screen Capture FPS: {fps}")



    def _rebuild_frame_buffer(self, region: Tuple[int, int, int, int]):
        frame_shape = (region[3] - region[1],region[2] - region[0],self.channel_size, )
        with self.__lock:
            self.__frame_buffer = cp.ndarray((self.max_buffer_len, *frame_shape), dtype=np.uint8)
            self.__head = 0
            self.__tail = 0
            self.__full = False

    def startN(
        self,
        region: Tuple[int, int, int, int] = None,
        target_fps: int = 0,
        video_mode=False,
    ):

        self.is_capturing = True
        frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
        self.__frame_buffer = np.ndarray(
            (self.max_buffer_len, *frame_shape), dtype=np.uint8
        )
        self.__thread = Thread(
            target=self.__captureN,
            name="DXCamera",
            args=(region, target_fps, video_mode),
        )
        self.__thread.daemon = True
        self.__thread.start()

    def get_latest_frameN(self):
        self.__frame_available.wait()
        #with self.__lock:
        if self.__frame_count > 0:
            ret = self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
            self.__frame_available.clear()
            self.__frame_count -= 1
            return np.array(ret)
        return None

    def __captureN(self, region: Tuple[int, int, int, int], target_fps: int = 0, video_mode=False):
        while not self.__stop_capture.is_set():
            try:
                frame = self._grab(region)
                if frame is not None:
                    with self.__lock:
                        self.__frame_buffer[self.__head] = frame
                        if self.__full:
                            self.__tail = (self.__tail + 1) % self.max_buffer_len
                        self.__head = (self.__head + 1) % self.max_buffer_len
                        self.__frame_available.set()
                        self.__frame_count += 1
                        self.__full = self.__head == self.__tail
                elif video_mode:
                    with self.__lock:
                        self.__frame_buffer[self.__head] = np.array(
                            self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
                        )
                        if self.__full:
                            self.__tail = (self.__tail + 1) % self.max_buffer_len
                        self.__head = (self.__head + 1) % self.max_buffer_len
                        self.__frame_available.set()
                        self.__frame_count += 1
                        self.__full = self.__head == self.__tail

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.__stop_capture.set()
                raise e

        fps = int(self.__frame_count / (time.perf_counter() - self.__capture_start_time))
        print(f"Screen Capture FPS: {fps}")



    def _rebuild_frame_bufferN(self, region: Tuple[int, int, int, int]):
        frame_shape = (region[3] - region[1],region[2] - region[0],self.channel_size, )
        with self.__lock:
            self.__frame_buffer = np.ndarray((self.max_buffer_len, *frame_shape), dtype=np.uint8)
            self.__head = 0
            self.__tail = 0
            self.__full = False


    def _validate_region(self, region: Tuple[int, int, int, int]):
        l, t, r, b = region
        if not (self.width >= r > l >= 0 and self.height >= b > t >= 0):
            raise ValueError(f"Invalid Region: Region should be in {self.width}x{self.height}")

    def release(self):
        self.stop()
        self._duplicator.release()
        self._stagesurf.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return "<{}:\n\t{},\n\t{},\n\t{},\n\t{}\n>".format(
            self.__class__.__name__,
            self._device,
            self._output,
            self._stagesurf,
            self._duplicator,
        )


