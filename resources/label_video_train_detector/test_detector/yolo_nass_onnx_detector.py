import cv2
import numpy as np
import onnxruntime
import onnx

import time


class DictWrapper(object):

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [DictWrapper(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, DictWrapper(b) if isinstance(b, dict) else b)


class YoloNASObjectDetector:

    def __init__(self, model_name, weights_name, min_conf, nms_threshold, args):
        providers = self.provider_by_device[args.device]
        self.session = onnxruntime.InferenceSession(weights_name,
                                                    providers=providers)
        self.inputs = [o.name for o in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.min_conf = min_conf
        self.classes = args.classes

    provider_by_device = {
        "CPU": ['CPUExecutionProvider'],
        "GPU": ['CUDAExecutionProvider'],
    }

    def detect(self, frame):
        original_image_shape = frame.shape
        image = self.resize_image(frame, (640, 640), True)
        resized_image_shape = image.shape
        h, w = image.shape[:2]
        if h != 640 or w != 640:
            image = np.pad(image, ((0, 640 - h), (0, 640 - w), (0, 0)),
                                   mode='constant', constant_values=0)
        image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

        num_detections, pred_boxes, pred_scores, pred_classes = self.session.run(self.outputs, {self.inputs[0]: image_bchw})

        def process_box(box):
            x_delta = original_image_shape[1] / resized_image_shape[1]
            y_delta = original_image_shape[0] / resized_image_shape[0]
            x1, y1, x2, y2 = int(box[0] * x_delta), int(box[1] * y_delta), int(box[2]*x_delta), int(box[3]*y_delta)
            return x1, y1, x2 - x1 + 1, y2 - y1 + 1

        # for image_index in range(num_detections.shape[0]):
        image_index = 0
        img_count = num_detections[image_index, 0]
        dets = [{
            "box": process_box(box),
            "conf": score,
            "class": int(klass)
        } for box, score, klass in
                       zip(pred_boxes[image_index][:img_count],
                           pred_scores[image_index][:img_count],
                           pred_classes[image_index][:img_count]) if klass in self.classes
            if score > self.min_conf]
        return dets

    def resize_image(self, image, size, keep_aspect_ratio=False):
        if not keep_aspect_ratio:
            resized_frame = cv2.resize(image, size)
        else:
            h, w = image.shape[:2]
            scale = min(size[1] / h, size[0] / w)
            resized_frame = cv2.resize(image, None, fx=scale, fy=scale)
        return resized_frame
