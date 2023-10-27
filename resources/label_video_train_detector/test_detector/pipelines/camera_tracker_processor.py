class CameraTrackerProcessor:

    def __init__(self, tracker, camera_name):
        self.camera_name = camera_name
        self.tracker = tracker

    def process(self, data):
        if 'cameras' in data:
            cam_data = data['cameras'].get(self.camera_name)
            if cam_data is None:
                return data
            detections = cam_data['detections']
            self.tracker.process(detections, data['frame_idx'])
            cam_data['tracked_objects'] = {t[2]:t for t in self.tracker.tracked_objects}
        return data
