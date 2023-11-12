import cv2
import numpy as np
import threading
import queue
import logging
import sys
import ultralytics.yolo.engine.model as myultra


class Yolo8Detector:

    def __init__(self, model_name, weights_name, min_conf, nms_threshold, args):
        self.nms_threshold = nms_threshold
        self.model = myultra.YOLO(weights_name)
        self.min_conf = min_conf
        self.classes = args.classes

    def detect(self, img):
        results = self.model.predict(img)
        return self.dets_from_yolov8_tracker(results)

    def dets_from_yolov8_tracker(self, yolov8_results):
        yolov8_results = yolov8_results[0]
        xyxy = yolov8_results.boxes.xyxy.cpu().numpy()
        confidence = yolov8_results.boxes.conf.cpu().numpy()
        class_id = yolov8_results.boxes.cls.cpu().numpy().astype(int)

        masks = []
        if yolov8_results.masks is not None:
            masks = yolov8_results.masks.xy

        keypoints = []
        if yolov8_results.keypoints is not None:
            keypoints = yolov8_results.keypoints.cpu().numpy()

        tracks_id = None
        if yolov8_results.boxes.id is not None:
            tracks_id = yolov8_results.boxes.id.cpu().numpy().astype(int)

        result = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = xyxy[i]
            result.append({
                "box": (int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)),
                "conf": confidence[i],
                "class": class_id[i],
                "class_name": yolov8_results.names[class_id[i]],
                "keypoints": keypoints[i] if len(keypoints) > 0 else [],
                "tracks_id": tracks_id[i] if tracks_id is not None else None,
                "masks": masks[i] if len(masks) > 0 else []
            })
        return result


K_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
           'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
           'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
           'left_ankle', 'right_ankle'] #it is set later in the code just in case
kpairs = [
    (0, 1), (1, 3), (0, 2), (2, 4), (0, 5), (0, 6),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
          (12, 14), (13, 15), (14, 16)]
pair_colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 128, 128),
    (255, 165, 0),
    (0, 128, 0),
    (128, 0, 0),
    (128, 0, 0),
    (0, 0, 128),
    (128, 255, 0),
    (255, 128, 0),
    (255, 0, 128),
    (128, 0, 255)
]

colors = []
for j in range(200):
    random_color=(int(np.random.choice(range(255))), int(np.random.choice(range(255))), int(np.random.choice(range(255))))
    colors.append(random_color)


class PaintYoloDetectionProcessor:
    def __init__(self, field_name, show_class=False, show_keypoints=True,
                 show_mask=False):
        self.show_mask = show_mask
        self.show_keypoints = show_keypoints
        self.show_class = show_class
        self.field_name = field_name

    def process(self, data):
        to_show = data.get(self.field_name)
        if to_show is not None:
            result = self.draw_tracks(to_show, data)
            data[self.field_name] = result
        return data

    def draw_tracks(self, frame, data):
        mask_img = frame.copy()
        for det in data['detections']:
            box = det['box']
            conf = det["conf"]
            class_name = det["class_name"]
            cls = det["class"]
            keyp = det.get('keypoints')
            mask = det.get('masks')

            label = f"{class_name}({conf * 100:.0f})"

            color_trk = colors[cls]
            if self.show_class:
                self.box_label(frame, box, label, color_trk)

            if self.show_keypoints and keyp is not None:
                # draw points
                for (x, y, score) in keyp:
                    cv2.circle(frame, (int(x), int(y)), 2, color_trk, 2)

                # draw lines between keypoints
                for idx, (p1, p2) in enumerate(kpairs):
                    (x1, y1, s1) = keyp[p1]
                    (x2, y2, s2) = keyp[p2]
                    color = pair_colors[idx]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            if self.show_mask and mask is not None:
                pts = []
                for coord in mask:
                    pts.append((int(coord[0]), int(coord[1])))
                points_array = np.array(pts)
                cv2.drawContours(mask_img, [points_array], -1, color_trk, -1)

        alpha = 0.5
        return cv2.addWeighted(frame, alpha, mask_img, 1-alpha, 0)

    def box_label(self, image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (box[0], box[1]), (box[0]+box[2], box[1]+box[3])
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)