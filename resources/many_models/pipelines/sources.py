from .ffmpeg_file_reader import FfmpegFileReader
import cv2


class VideoBasedSource:
    def __init__(self, file_name, start_frame_idx=0):
        self.readers = []
        self.next_frame_idx = start_frame_idx
        self.reader = FfmpegFileReader(file_name, (1024, -1), 15, start_frame_idx)
        self.reader.start()

    def get_data(self):
        success, image, time = self.reader.read()
        if success:
            data = {'image': image, 'time': time, 'success': success, 'frame_idx': self.next_frame_idx}
        else:
            data = {'stop_pipeline': True}
        self.next_frame_idx += 1
        return data


class OpencvVideoSource:
    def __init__(self, file_name, start_frame_idx=0):
        self.reader = cv2.VideoCapture(file_name)
        self.next_frame_idx = start_frame_idx
        if start_frame_idx > 0:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx - 1)

    def get_data(self):
        res, frame = self.reader.read()
        if res:
            data = {'image': frame, 'frame_idx': self.next_frame_idx}
        else:
            data = {'stop_pipeline': True}
        self.next_frame_idx += 1
        return data
