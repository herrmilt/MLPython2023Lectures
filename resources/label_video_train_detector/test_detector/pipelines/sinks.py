import cv2
import numpy as np

from .ffmpeg_file_writer import FfmpegFileWriter
import time


class ShowCamerasInWindowsSink:
    def __init__(self, field_name, window_caption="window"):
        self.field_name = field_name
        self.window_caption = window_caption

    def put_data(self, data):
        if 'cameras' in data:
            for cam_name, cam_values in data['cameras'].items():
                to_show = cam_values[self.field_name]
                cv2.imshow(self.window_caption + "_" + cam_name, to_show)


class WaitForKeySink:
    def __init__(self, wait_time=0):
        self.wait_time = wait_time

    def put_data(self, data):
        if 'cameras' in data:
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


class WriteCamerasToVideoSink:
    def __init__(self, field_name, result_name_pattern, fps):
        self.field_name = field_name
        self.fps = fps
        self.result_name_pattern = result_name_pattern
        self.file_writers = {}
        self.add_row = False
        self.row_width = None

    def put_data(self, data):
        if 'cameras' in data:
            for cam_name, cam_values in data['cameras'].items():
                image = cam_values.get(self.field_name)
                if image is None:
                    continue
                writer = self.file_writers.get(cam_name)
                if not writer:
                    h, w, _ = image.shape
                    if h % 2 == 1:
                        self.add_row = True
                        self.row_width = w
                    writer = FfmpegFileWriter(self.result_name_pattern.format(cam_name), w,
                                              h if not self.add_row else h+1, self.fps)
                    writer.open()
                    self.file_writers[cam_name] = writer
                if self.add_row:
                    image = np.vstack([image, np.zeros((1, self.row_width, 3))])
                writer.write(image)

    def stop(self):
        for w in self.file_writers.values():
            w.close()


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
