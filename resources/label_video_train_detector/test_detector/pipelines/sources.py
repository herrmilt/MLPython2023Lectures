from .ffmpeg_file_reader import FfmpegFileReader


class MediaServerSource:

    def __init__(self, video_stream, start_frame_idx=0):
        self.video_stream = video_stream
        self.next_frame_idx = start_frame_idx

    def start(self):
        self.video_stream.start()

    def stop(self):
        self.video_stream.stop()

    def get_data(self):
        self.video_stream.next()
        all_images = self.video_stream.read_all()
        data = {'cameras': all_images, 'frame_idx': self.next_frame_idx}
        self.next_frame_idx += 1
        return data


class VideoBasedSource:
    def __init__(self, pairs, start_frame_idx=0):
        self.readers = []
        self.next_frame_idx = start_frame_idx
        for file_name, cam_name in pairs:
            reader = FfmpegFileReader(file_name, (1024, -1), 15, start_frame_idx)
            reader.start()
            self.readers.append((cam_name, reader))

    def get_data(self):
        all_images = {}
        some_success = False
        for cam_name, reader in self.readers:
            success, image, time = reader.read(cam_name)
            some_success = some_success or success
            all_images[cam_name] = {'image': image, 'time': time, 'success': success}
        if some_success:
            data = {'cameras': all_images, 'frame_idx': self.next_frame_idx}
        else:
            data = {'stop_pipeline': True}
        self.next_frame_idx += 1
        return data
