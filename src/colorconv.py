'''This module contains the implementation of the ConvexCluster class.'''

import random
import numpy as np
from scipy.spatial import (cKDTree,
                           ConvexHull)


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


class ConvexCluster:
    '''ConvexCluster class'''

    def __init__(self, X, d, N, s, c, coords, reverse_labels):
        self.X = X # grid points
        self.d = d # intial distance
        self.N = N # number of categories
        self.s = s # smoothing factor
        self.c = c # convexity factor
        self.labels = {} # list of points per category
        self.reverse_labels = reverse_labels

        # initialize the grid
        # self.coords = [(i, j) for i in range(X.shape[0]) 
        #                for j in range(X.shape[1])]
        self.coords = coords
        self.tree = cKDTree(self.coords)
        # random choice N points
        self.centroids = random.sample(self.coords, N)
    
    def point_in_hull(self, point, hull, tolerance=1e-12):
        '''Check if a point is inside a convex hull'''
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
            for eq in hull.equations)
    
    def initial_partition(self):
        '''Initialize the partition by assigning each point to the closest category'''
        for i, c in enumerate(self.centroids):
            self.labels[i] = [c]
        
        # for each centroid, find the closest points within distance d 
        # and assign them to the same category
        for i, c in enumerate(self.centroids):
            _, ind = self.tree.query(c, k=len(self.coords), 
                                            distance_upper_bound=self.d)
            for j in ind:
                if j < len(self.coords):
                    self.labels[i].append(self.coords[j])
    
    def update_partition(self):
        '''Update the partition by assigning each point to the closest category'''
        # get a list of points that are unlabeled
        unlabeled = [p for p in self.coords if 
                     p not in sum(self.labels.values(), [])]
        while unlabeled:
            # randomly sample one of the unlabeled points
            p = random.choice(unlabeled)
            # find the distance to each of the closest points in each category
            distances = []
            for _, l in self.labels.items():
                distances.append(min([np.linalg.norm(np.array(p) - np.array(q)) 
                                      for q in l]))
            # apply softmax to the distances
            probs = softmax(1 / np.array(distances), temperature=self.c)
            # choose the category with the highest probability
            i = np.random.choice(range(len(self.labels)), p=probs)
            # add the point to the category
            self.labels[i].append(p)
            # remove the point from the unlabeled list
            unlabeled.remove(p)

    def update_centroids(self):
        '''Update the centroids by taking the mean of the points in each category'''
        for i, l in self.labels.items():
            try:
                hull = ConvexHull(l)
                # find points that are in the convex hull
                # but do not belong to the category
                in_hull = [p for p in self.coords if 
                        self.point_in_hull(p, hull)]
                for p in random.sample(in_hull, 
                                    int(self.s * len(in_hull))):
                    # assign the point to the category
                    # remove the point from the previous category
                    for j, l in self.labels.items():
                        if p in l:
                            self.labels[j].remove(p)
                    self.labels[i].append(p)
            except:
                pass

    def degree_of_convexity_cluster(self, cluster_points):
        '''
        Build the convex hull around cluster. 
        Find the number of points in the hull. 
        Divide the number of points in the class 
        by the number of points inside the hull
        '''
        try:
            hull = ConvexHull(cluster_points)
            in_hull = [p for p in self.coords if 
                    self.point_in_hull(p, hull)]
            return len(cluster_points) / (len(in_hull) + 1)
        except:
            return 1

    def degree_of_convexity(self):
        '''The number of labels is X times X + N ???'''
        return np.average([self.degree_of_convexity_cluster(cluster_points)
                    for cluster_points in self.labels.values()],
                    weights=[len(cluster_points) 
                    for cluster_points in self.labels.values()])

    def run(self):
        '''Run the algorithm. Return the label matrix.'''
        self.initial_partition()
        self.update_partition()
        self.update_centroids()

        # real_labels =  [(i, j) for i in range(X.shape[0]) 
        #                for j in range(X.shape[1])]
        
        label_matrix = np.zeros(self.X.shape)
        for i, l in iter(self.labels.items()):
            for p in l:
                label_matrix[self.reverse_labels[p]] = i
        return label_matrix
