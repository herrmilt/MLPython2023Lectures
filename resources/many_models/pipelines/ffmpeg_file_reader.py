import datetime as dt
import os.path
from threading import Thread

import numpy
import subprocess as sp
import cv2
import re


class FfmpegFileReader:

    ffmpeg_stderr = open("../ffmpeg_sterr.log", "a")

    def __init__(self, source_in, frame_size, fps, debug_start_frame, gray_scale=False,
                 use_tcp=True, gpu_accel=False):
        original_width, original_height = None, None
        # w_h = self.get_video_resolution(source_in)
        # if w_h:
        #     original_width, original_height = w_h
        # if not w_h or original_width == 0:
        if True:
            cap = cv2.VideoCapture(source_in)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        desired_width = frame_size[0]
        desired_height = round((original_height / original_width) * desired_width)
        self.gpu_accel = gpu_accel
        self.is_file = os.path.exists(source_in)
        if self.is_file:
            self.start_time = dt.datetime.now()
        self.use_tcp = use_tcp and not self.is_file
        self.input_source = source_in
        self.frames_w = desired_width
        self.frames_h = desired_height
        self.FFMPEG_BIN = "/usr/bin/ffmpeg"
        # self.FFMPEG_BIN = "/mnt/data/__Software/ffmpeg_cuda/ffmpeg/ffmpeg"
        self.grabbed = False
        self.frame = None
        self.name = source_in
        self.chan = 3
        self.stopped = False
        self.frame_idx = -1
        self.fps = fps
        if gray_scale:
            self.chan = 1
        command = ([self.FFMPEG_BIN] +
                   (['-rtsp_transport', 'tcp'] if self.use_tcp else []) +
                   (['-hwaccel', 'cuda'] if self.gpu_accel else []) +
                   (['-ss', str(dt.timedelta(seconds=debug_start_frame // fps))] if debug_start_frame != 0 else []) +
                   ['-i', self.input_source] +
                   (["-c:v", "h264_nvenc"] if self.gpu_accel else []) +
                   ['-f', 'image2pipe',
                    '-vf', f'scale={frame_size[0]}:-1:force_original_aspect_ratio=decrease',
                    '-pix_fmt', 'bgr24',
                    '-r', str(self.fps),
                    '-vcodec', 'rawvideo', '-'])
        self.pipe = sp.Popen(command, stdout=sp.PIPE, stderr=FfmpegFileReader.ffmpeg_stderr, bufsize=10 ** 8)

    def get_video_resolution(self, source_in):
        command = [
            'ffmpeg',
            '-i', source_in,
            '-f', 'null',  # Null output format
            '-'
        ]
        result = sp.run(command, stderr=sp.PIPE, text=True)

        # Parse the output to extract video size
        output = result.stderr
        match = re.search(r"Stream #\d:\d\(\w+\): Video: .+ (\d+)x(\d+)", output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width, height
        else:
            return None

    def read(self):
        # read 420*360*3 bytes (= 1 frame)
        # start_time = datetime.datetime.now()
        raw_image = self.pipe.stdout.read(self.frames_w * self.frames_h * self.chan)
        if len(raw_image) != (self.frames_h * self.frames_w * self.chan):
            if self.is_file:
                return False, None, None
        self.frame_idx += 1
        # transform the byte read into a numpy array
        image = numpy.fromstring(raw_image, dtype='uint8')
        try:
            image = image.reshape((self.frames_h, self.frames_w, self.chan))
        except:
            image = None
        if self.is_file:
            current_time = self.start_time + dt.timedelta(seconds=self.frame_idx / self.fps)
        else:
            current_time = dt.datetime.now()
        return True, image, current_time

    def start(self):
        # start the thread to read frames from the video stream
        # t = Thread(target=self.update, name=self.name, args=())
        # t.daemon = True
        # t.start()
        return self

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
