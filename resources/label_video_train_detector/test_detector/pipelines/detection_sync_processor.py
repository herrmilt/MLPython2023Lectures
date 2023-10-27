class GetDetectionProcessor:

    def __init__(self, detector):
        self.detector = detector

    def process(self, data):
        if 'cameras' in data:
            for cam_name, d in data['cameras'].items():
                image = d['image']
                detections = self.detector.detect(image)
                d['detections'] = detections
        return data