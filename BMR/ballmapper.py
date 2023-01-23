import numpy as np
import random


class BallMapper:
    def __init__(self, points, epsilon, shuffle=False, order=None):

        # find vertices
        self.balls = {}  # dict of points {idx_b: idx_p, ... }
        self.epsilon = epsilon

        centers_counter = 0

        if order is None:
            order = list(range(len(points)))
            if shuffle:
                random.shuffle(order)

        for idx_p in order:
            # current point
            p = points[idx_p]
            is_covered = False

            for idx_b in self.balls:
                distance = np.linalg.norm(p - self.balls[idx_b]['position'])
                if distance <= self.epsilon:
                    is_covered = True
                    self.balls[idx_b]['points_covered'].append(idx_p)
                    break

            if not is_covered:
                self.balls[centers_counter] = {'idxp': idx_p,
                                               'position': points[idx_p],
                                               'points_covered': [],
                                               'model': None,
                                               'merged': [],
                                               }
                centers_counter += 1


    def find_balls(self, points, nearest_neighbour_extrapolation=False):
        """
        For each point in points returns a list containing ids for vertices(=ball) which distance
        to given point is lower than radius.

        :param points:
        :param nearest_neighbour_extrapolation:
        :return:
        """
        vertices_ids = []
        for p in points:
            vertices_ids_p = []
            covered = False
            min_distance = np.Inf
            min_distance_vert = None
            for idx_v in self.balls:
                distance = np.linalg.norm(p - self.balls[idx_v]['position'])
                if distance <= self.epsilon:
                    covered = True
                    vertices_ids_p.append(idx_v)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_vert = idx_v
            if not covered:
                if nearest_neighbour_extrapolation:
                    vertices_ids_p.append(min_distance_vert)
                else:
                    vertices_ids_p.append(None)
            vertices_ids.append(vertices_ids_p)
        return vertices_ids
