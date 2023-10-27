import cv2
import numpy as np
import subprocess as sp
import shlex


class FfmpegFileWriter:

    ffmpeg_stderr = open("../ffmpeg_sterr_write.log", "a")

    def __init__(self, file_name, width, height, fps, codec=None):
        self.codec = codec if codec is not None else "h264"
        self.fps = fps
        self.height = height
        self.width = width
        self.file_name = file_name
        self.process = None

    def open(self):
        self.process = sp.Popen(shlex.split(f'ffmpeg -y -s {self.width}x{self.height} -pixel_format bgr24 -f rawvideo -r {self.fps} -i pipe: -vcodec {self.codec} -pix_fmt yuv420p -crf 24 {self.file_name}'),
                                stdin=sp.PIPE, stdout=FfmpegFileWriter.ffmpeg_stderr, stderr=FfmpegFileWriter.ffmpeg_stderr)

    def write(self, image):
        if self.process is None:
            raise Exception("ERROR. Cannot write to an uninitialized writer.")
        self.process.stdin.write(image.tobytes())

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            # Wait for sub-process to finish
            self.process.wait()
            # Terminate the sub-process
            self.process.terminate()