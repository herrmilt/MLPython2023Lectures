from scipy.optimize import linear_sum_assignment
import numpy as np


class HungarianMatcherGlobal:
    def __init__(self, max_value):
        self.max_value = max_value

    def match(self, points1, points2, comp_function, max_value=None):
        g = self.calculate_similarity_matrix(points1, points2, comp_function)
        return self.apply_hungarian(g, len(points1), len(points2), max_value)

    def calculate_similarity_matrix(self, boxes1, boxes2, comp_function):
        G = np.zeros([len(boxes1), len(boxes2)])
        for idx1, box1 in enumerate(boxes1):
            for idx2, box2 in enumerate(boxes2):
                value = comp_function(box1, box2)
                # if value > self.max_value:
                #     value = 100
                G[idx1, idx2] = value
        return G

    def apply_hungarian(self, G, len1, len2, max_value=None):
        if not isinstance(G.shape, int):
            if max_value is None:
                max_value = self.max_value
            row_idxs, col_idxs = linear_sum_assignment(G)
            keep = []
            keep_person = set()
            keep_det = set()
            for row_idx, col_idx in zip(row_idxs, col_idxs):
                iou_value = G[row_idx, col_idx]
                if iou_value >= max_value:
                    continue
                keep.append((row_idx, col_idx))
                keep_person.add(row_idx)
                keep_det.add(col_idx)
            lost = [idx for idx in range(len1) if idx not in keep_person]
            appear = [idx for idx in range(len2) if idx not in keep_det]
            return lost, keep, appear

        else:
            return [idx for idx in range(len1)], [], [idx for idx in range(len2)]


class HungarianMatcherGlobalWithDummies:
    def __init__(self, max_value):
        self.max_value = max_value

    def match(self, points1, points2, comp_function, dummy_count, dummy_value, max_value=None):
        g = self.calculate_similarity_matrix(points1, points2, comp_function)
        r, c = g.shape
        dummy_g = np.full((r+dummy_count, c+dummy_count), dummy_value)
        dummy_g[:r, :c] = g
        return self.apply_hungarian(dummy_g, len(points1), len(points2), max_value, r, c)

    def calculate_similarity_matrix(self, boxes1, boxes2, comp_function):
        G = np.zeros([len(boxes1), len(boxes2)])
        for idx1, box1 in enumerate(boxes1):
            for idx2, box2 in enumerate(boxes2):
                value = comp_function(box1, box2)
                G[idx1, idx2] = value
        return G

    def apply_hungarian(self, G, len1, len2, max_value, r_count, c_count):
        if not isinstance(G.shape, int):
            if max_value is None:
                max_value = self.max_value
            row_idxs, col_idxs = linear_sum_assignment(G)
            keep = []
            keep_person = set()
            keep_det = set()
            for row_idx, col_idx in zip(row_idxs, col_idxs):
                iou_value = G[row_idx, col_idx]
                if iou_value >= max_value or row_idx >= r_count or col_idx >= c_count:
                    continue
                keep.append((row_idx, col_idx))
                keep_person.add(row_idx)
                keep_det.add(col_idx)
            lost = [idx for idx in range(len1) if idx not in keep_person]
            appear = [idx for idx in range(len2) if idx not in keep_det]
            return lost, keep, appear

        else:
            return [idx for idx in range(len1)], [], [idx for idx in range(len2)]


if __name__ == "__main__":
    m = np.array([[1, 2], [2, 1]])
    r, c = m.shape
    matcher = HungarianMatcherGlobalWithDummies(3)
    print(matcher.match(list(range(r)), list(range(c)), lambda p1, p2: m[p1, p2], 1, 2.5))