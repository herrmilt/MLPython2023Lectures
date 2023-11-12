import cv2
import random


def create_all_colors(count):
    random.seed(3146)
    return [(int(random.randint(50, 255)),
             int(random.randint(50, 255)),
             int(random.randint(50, 255)))
            for _ in range(count)]


all_colors = create_all_colors(100)


class CopyImageProcessor:
    def __init__(self, source, dest):
        self.dest = dest
        self.source = source

    def process(self, data):
        if self.source in data:
            data[self.dest] = data[self.source].copy()
        return data


class PaintDetectionsProcessor:
    def __init__(self, field_name, show_class=False):
        self.show_class = show_class
        self.field_name = field_name

    def process(self, data):
        to_show = data.get(self.field_name)
        if to_show is not None:
            self.paint_detections(to_show, data)
            data[self.field_name] = to_show
        return data

    def paint_detections(self, to_show, values):
        for det in values['detections']:
            box = det['box']
            conf = det['conf']
            cv2.rectangle(to_show, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (255, 255, 0), 2)
            if self.show_class:
                class_name = det.get('class_name', det.get('class', ''))
                text = f"{class_name}({conf * 100:.0f})"
            else:
                text = f"{conf*100:.0f}"
            cv2.putText(to_show, text, (box[0], box[1]-3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)


class PaintTrackers:

    def __init__(self, field_name):
        self.field_name = field_name

    def process(self, data):
        tracked_objects = data.get('tracked_objects')
        if tracked_objects is not None:
            to_show = data[self.field_name]
            for (x, y, w, h), conf, t_id, det in tracked_objects.values():
                color = all_colors[t_id % 100]
                caption = str(t_id)

                cv2.rectangle(to_show, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(to_show, caption, (int(x), int(y) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness=2)
            data[self.field_name] = to_show
        return data


class PaintFrameId:
    def __init__(self, field_name):
        self.field_name = field_name

    def process(self, data):
        if 'frame_idx' in data and 'cameras' in data:
            msg = f"frm:{data['frame_idx']}"
            for cam_name, cam_values in data['cameras'].items():
                to_show = cam_values[self.field_name]
                h, w, _ = to_show.shape
                cv2.putText(to_show, msg, (0, h-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

        return data