import cv2
import numpy as np

from .ffmpeg_file_writer import FfmpegFileWriter
import time


class ShowInWindowsSink:
    def __init__(self, field_name, window_caption="window"):
        self.field_name = field_name
        self.window_caption = window_caption

    def put_data(self, data):
        to_show = data.get(self.field_name)
        if to_show is not None:
            cv2.imshow(self.window_caption, to_show)


class WaitForKeySink:
    def __init__(self, wait_time=0):
        self.wait_time = wait_time

    def put_data(self, data):
        key = cv2.waitKey(self.wait_time)
        if key == ord(" "):
            self.wait_time = 1 - self.wait_time


class FpsCalculator:
    def __init__(self):
        self.fps_sum = 0
        self.fps_count = 0
        self.previous_time = None

    def put_data(self, data):
        now_time = time.time()
        if self.previous_time is not None:
            fps = 1 / (now_time - self.previous_time + 0.000001)
            self.fps_sum += fps
            self.fps_count += 1
        self.previous_time = now_time

    def stop(self):
        print(f"Average FPS: {self.fps_sum / self.fps_count}")


class ShowCameraSink:
    def __init__(self, field_name, window_caption="window"):
        self.field_name = field_name
        self.window_caption = window_caption

    def put_data(self, data):
        if self.field_name in data:
            to_show = data[self.field_name]
            cv2.imshow(self.window_caption, to_show)


class WriteImageToVideoSink:
    def __init__(self, field_name, result_name, fps):
        self.field_name = field_name
        self.fps = fps
        self.result_name = result_name
        self.file_writer = None

    def put_data(self, data):
        if self.field_name in data:
            image = data[self.field_name]
            if not self.file_writer:
                h, w, _ = image.shape
                self.file_writer = FfmpegFileWriter(self.result_name, w, h, self.fps)
                self.file_writer.open()
            self.file_writer.write(image)

    def stop(self):
        self.file_writer.close()
