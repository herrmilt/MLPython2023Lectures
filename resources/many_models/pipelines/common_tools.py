import datetime
import logging
import math
from collections import defaultdict

import cv2
import numpy as np
import functools
import time
from numba import jit

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


@jit(nopython=True)
def cosine_distance_numba(u: np.ndarray, v: np.ndarray):
    if u is None or v is None:
        return 1
    assert (u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / (np.sqrt(uu) * np.sqrt(vv))
    return 1 - cos_theta


def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2]) * (boxA[3])
    boxBArea = (boxB[2]) * (boxB[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = max(0., interArea / float(boxAArea + boxBArea - interArea))
    # return the intersection over union value
    return iou


def calc_ios(source, other):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(source[0], other[0])
    yA = max(source[1], other[1])
    xB = min(source[0] + source[2], other[0] + other[2])
    yB = min(source[1] + source[3], other[1] + other[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (source[2]) * (source[3])
    boxBArea = (other[2]) * (other[3])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea)
    # return the intersection over union value
    return iou


def get_box_center(box):
    return get_box_relative(box, 0.5, 0.5)


def get_box_relative(box, rx, ry):
    return box[0] + int(box[2]*rx), box[1] + int(box[3]*ry)


def point_dist(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def extract_subimage(image, box):
    return image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]


def fix_box(box, width, height):
    x, y, w, h = box
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    w = min(w, width - x - 1)
    h = min(h, height - y - 1)
    return x, y, w, h


def enlarge_box(box, delta, image_width, image_height):
    return fix_box([box[0] - delta, box[1] - delta, box[2] + 2 * delta, box[3] + 2 * delta], image_width, image_height)


def calc_box_bottom(box, delta):
    return box[0], box[1] + int(box[3] * (1-delta)), box[2], int(box[3]*delta)


def arg_max(values):
    return max(range(len(values)), key=lambda i: values[i])


def arg_min(values):
    return min(range(len(values)), key=lambda i: values[i])


def boxes_distance(a, b):
    a_x1 = a[0]
    a_x2 = a[0] + a[2]
    a_y1 = a[1]
    a_y2 = a[1] + a[3]
    b_x1 = b[0]
    b_x2 = b[0] + b[2]
    b_y1 = b[1]
    b_y2 = b[1] + b[3]
    A_min = np.array([a_x1, a_y1])
    A_max = np.array([a_x2, a_y2])
    B_min = np.array([b_x1, b_y1])
    B_max = np.array([b_x2, b_y2])
    delta1 = A_min - B_max
    delta2 = B_min - A_max
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist


def drawpoly(img, polygon, color, thickness=1, gap=5, map_point=lambda p: p):
    for p1, p2 in zip(polygon, polygon[1:]):
        drawline(img, map_point(p1), map_point(p2), color, thickness, gap)


def drawline(img, pt1, pt2, color, thickness=1, gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    for idx, p in enumerate(pts):
        cv2.circle(img, p, radius=thickness, color=color)


class ImageChangeDetector:

    def __init__(self, min_threshold=3):
        self.min_threshold = min_threshold
        self.prev_image = None

    def update_image(self, image):
        if self.prev_image is None:
            # self.prev_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.prev_image = image
            return True
        else:
            # current_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_image = image
            dif = cv2.absdiff(self.prev_image, current_image)
            mean, stdev = cv2.meanStdDev(dif)
            self.prev_image = current_image
            stdev = max(stdev)[0]
            return stdev > self.min_threshold


class DictWrapper(object):

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [DictWrapper(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, DictWrapper(b) if isinstance(b, dict) else b)


class Timer:

    def __init__(self, name=None, active=True):
        self.name = name
        self.active = active
        self.start = None

    def __enter__(self):
        if self.active:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.active:
            end = time.perf_counter()
            logging.debug(f"Time running {self.name}: {(end-self.start)*1000:.2f}ms")


class TimerConsole:

    def __init__(self, name=None, active=True):
        self.name = name
        self.active = active
        self.start = None

    def __enter__(self):
        if self.active:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        if self.active:
            end = time.perf_counter()
            print(f"Time running {self.name}: {(end-self.start)*1000:.2f}ms")


class TimerExt:

    all_times = {}
    are_new_fields = False
    field_name_order = []
    file_handler = None

    @staticmethod
    def set_output_file_name(file_name):
        TimerExt.file_handler = open(file_name, "a")
        TimerExt.file_handler.write(f"Session started at: {datetime.datetime.now()}\n")

    @staticmethod
    def add_value(value_name, value):
        if value_name in TimerExt.all_times:
            TimerExt.all_times[value_name] += value
        else:
            TimerExt.all_times[value_name] = value
            TimerExt.field_name_order.append(value_name)

    def __init__(self, name=None, is_main=False):
        self.is_main = is_main
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        end = time.perf_counter()
        time_delta = (end-self.start)*1000
        if self.name not in TimerExt.all_times:
            TimerExt.are_new_fields = True
            TimerExt.all_times[self.name] = time_delta
            TimerExt.field_name_order.append(self.name)
        else:
            TimerExt.all_times[self.name] += time_delta
        if self.is_main:
            if TimerExt.are_new_fields:
                TimerExt.file_handler.write(",".join(TimerExt.field_name_order) + "\n")
            TimerExt.file_handler.write(",".join((f"{TimerExt.all_times[field_name]:.1f}" for field_name in TimerExt.field_name_order))
                    + "\n")
            for f_name in TimerExt.field_name_order:
                TimerExt.all_times[f_name] = 0
            TimerExt.are_new_fields = False
            # TimerExt.file_handler.flush()
