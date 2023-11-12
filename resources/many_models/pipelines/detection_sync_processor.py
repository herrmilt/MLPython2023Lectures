class GetDetectionProcessor:

    def __init__(self, detector):
        self.detector = detector

    def process(self, data):
        image = data.get('image')
        if image is not None:
            detections = self.detector.detect(image)
            data['detections'] = detections
        return data
