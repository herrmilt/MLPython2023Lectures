from .hungarian_matcher_global import HungarianMatcherGlobalWithDummies, HungarianMatcherGlobal
from .common_tools import calc_iou, point_dist


class SimpleObjectTracker:

    class TrackedObject:

        def __init__(self, track_id, box, conf, start_frame_idx, data):
            self.track_id = track_id
            self.box = box
            self.conf = conf
            self.frame_idx = start_frame_idx
            self.update_count = 1
            self.missing_count = 0
            self.data = data

        def update(self, det, frame_idx):
            self.box = det['box']
            self.conf = det['conf']
            self.frame_idx = frame_idx
            self.update_count += 1
            self.missing_count = 0
            self.data = det

    def __init__(self, min_compare_value=0.25):
        self.matcher = HungarianMatcherGlobal(min_compare_value)
        self.next_track_id = 0
        self.__tracked_objects = []
        self.min_conf_first_object = 0.5
        self.min_count_trackers = 3
        self.max_missing_count = 0
        self.max_dist_same_point = 15

    @property
    def tracked_objects(self):
        return [(o.box, o.conf, o.track_id, o.data) for o in self.__tracked_objects
                if o.update_count >= self.min_count_trackers]

    def process(self, detections, frame_idx):
        missing_tracked_obj_idxs, keep_idxs, new_detection_idxs = self.matcher.match(self.__tracked_objects,
                                                                              detections,
                                                                              self.compare_fn, 0.5)
        for dets in (detections[idx] for idx in new_detection_idxs):
            self.check_and_create(dets['box'], dets['conf'], frame_idx, dets)
        for idx_track, idx_det in keep_idxs:
            self.__tracked_objects[idx_track].update(detections[idx_det], frame_idx)
        for idx_det in missing_tracked_obj_idxs:
            if self.__tracked_objects[idx_det].missing_count >= self.max_missing_count:
                self.__tracked_objects[idx_det] = None
            else:
                self.__tracked_objects[idx_det].missing_count += 1
        self.__tracked_objects = [t for t in self.__tracked_objects if t is not None]

    def compare_fn(self, o1, o2):
        return 1 - calc_iou(o1.box, o2['box'])

    def check_and_create(self, box, conf, frame_idx, data):
        if conf < self.min_conf_first_object:
            return
        box_points = self.get_box_points(box)
        points = [self.get_box_points(t.box) for t in self.__tracked_objects]
        points_match = max([sum(((point_dist(p1, p2) < self.max_dist_same_point)
                                 for p1, p2 in zip(p, box_points)))
                            for p in points],
                           default=0)
        if points_match >= 1:
            return
        self.__tracked_objects.append(SimpleObjectTracker.TrackedObject(
            self.next_track_id, box, conf, frame_idx, data))
        self.next_track_id += 1

    def get_box_points(self, box):
        return (box[0], box[1]), (box[0], box[1]+box[3]), (box[0]+box[2], box[1]+box[3]), (box[0]+box[2], box[1])