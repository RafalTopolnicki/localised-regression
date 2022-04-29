import numpy as np
import networkx as nx
import random
from tqdm.notebook import tqdm


class BallMapper:
    def __init__(self, points, coloring_df, epsilon, shuffle=False, order=None):
        # find vertices
        self.vertices = {} # dict of points {idx_v: idx_p, ... }
        self.vertices_pos = {} # dict of actual vertices locations - ball centers
        self.epsilon = epsilon

        centers_counter = 1
        
        if order is None:
            order = list(range(len(points)))
            if shuffle:
                random.shuffle(order)

        for idx_p in order:
            # current point
            p = points[idx_p]
            is_covered = False

            for idx_v in self.vertices:
                distance = np.linalg.norm(p - points[self.vertices[idx_v]])
                if distance <= self.epsilon:
                    is_covered = True
                    break

            if not is_covered:
                self.vertices[centers_counter] = idx_p
                self.vertices_pos[centers_counter] = points[idx_p]
                centers_counter += 1
                    
        # compute points_covered_by_landmarks
        self.points_covered_by_landmarks = dict()
        for idx_v in self.vertices:
            self.points_covered_by_landmarks[idx_v] = []
            for idx_p in order:
                distance = np.linalg.norm(points[idx_p] - points[self.vertices[idx_v]])
                if distance <= self.epsilon:
                    self.points_covered_by_landmarks[idx_v].append(idx_p)
                
        # find edges
        self.edges = [] # list of edges [[idx_v, idx_u], ...]
        for i, idx_v in tqdm(enumerate(list(self.vertices.keys())[:-1]), disable=True):
            for idx_u in list(self.vertices.keys())[i+1:]:
                if len(set(self.points_covered_by_landmarks[idx_v]).intersection(self.points_covered_by_landmarks[idx_u])) != 0:
                    self.edges.append([idx_v, idx_u])
                

                    
        # create Ball Mapper graph
        self.Graph = nx.Graph()
        self.Graph.add_nodes_from(self.vertices)
        self.Graph.add_edges_from(self.edges)
        
        MIN_SCALE = 100
        MAX_SCALE = 500
    
        MAX_NODE_SIZE = max([len(self.points_covered_by_landmarks[key]) 
                                 for key in self.points_covered_by_landmarks])
       
        for node in self.Graph.nodes:
            self.Graph.nodes[node]['points covered'] = self.points_covered_by_landmarks[node]
            self.Graph.nodes[node]['beta'] = None
            self.Graph.nodes[node]['intercept'] = None
            self.Graph.nodes[node]['size'] = len(self.Graph.nodes[node]['points covered'])
            # rescale the size for display
            self.Graph.nodes[node]['size rescaled'] = MAX_SCALE*self.Graph.nodes[node]['size']/MAX_NODE_SIZE + MIN_SCALE 
            self.Graph.nodes[node]['color'] = 'r'
            
            for name, avg in coloring_df.loc[self.Graph.nodes[node]['points covered']].mean().iteritems():
                self.Graph.nodes[node][name] = avg
        
    def find_balls(self, points, nearest_neighbour_extrapolation=False):
        vertices_ids = []
        for p in points:
            vertices_ids_p = []
            covered = False
            min_distance = np.Inf
            min_distance_vert = None
            for idx_v in self.vertices:
                distance = np.linalg.norm(p - self.vertices_pos[idx_v])
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
