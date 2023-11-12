class CameraTrackerProcessor:

    def __init__(self, tracker):
        self.tracker = tracker

    def process(self, data):
        detections = data.get('detections')
        if detections is not None:
            self.tracker.process(detections, data['frame_idx'])
            data['tracked_objects'] = {t[2]:t for t in self.tracker.tracked_objects}
        return data
