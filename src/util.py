import gzip
import numpy as np
from scipy.spatial import ConvexHull
from numcompress import compress

def safelog(x):
    '''
    Compute the logarithm of a positive number x, or return 0 if x is non-positive.

    Parameters:
        x (float): The number to compute the logarithm of.

    Returns:
        float: The logarithm of x if x is positive, or 0 if x is non-positive.
    '''
    return np.log(x) if x > 0 else 0


def non_dominated_front(points_x, points_y):
    """
    Returns the non-dominated (Pareto) front of a list of 2-D points using Kung's algorithm.
    
    Args:
    - points_x: List or array containing the x-coordinates of points.
    - points_y: List or array containing the y-coordinates of points.
    
    Returns:
    - List of indices representing the points on the Pareto front.
    """
    assert len(points_x) == len(points_y), "Number of x and y points must be the same"
    
    n = len(points_x)
    pareto_front = []
    dominated = [False] * n
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if points_x[i] <= points_x[j] and points_y[i] <= points_y[j]:
                    if points_x[i] < points_x[j] or points_y[i] < points_y[j]:
                        dominated[j] = True
    
    for i in range(n):
        if not dominated[i]:
            pareto_front.append(i)
    
    return pareto_front

def compression(grid):
    """
    Compresses the given grid using zlib compression algorithm.

    Parameters:
    grid (numpy.ndarray): The grid to be compressed.

    Returns:
    int: The length of the compressed grid in bytes.
    """
    return np.log(len(gzip.compress(grid.tobytes(), compresslevel=9)))

def cost(grid, p_m, cielab_dict):
    '''
    Cost function from Carr et al. (2020) for the informativeness of a grid.

    Parameters:
        grid (numpy.ndarray): The grid to compute the cost of.
        p_m (numpy.ndarray): The prior over each cell in the grid.
        cielab_dict (dict): A dictionary mapping the coordinates of each cell to its CIELAB color.
    
    Returns:
        float: The cost of the grid.

    '''
    # get all unique values in the grid
    C = np.unique(grid)
    # get list of all coordinates of each cell in the grid
    M = [(i, j) for i in range(grid.shape[0]) 
                       for j in range(grid.shape[1])]
    # p_m equals to 1 divided by the number of cells in the grid
    if p_m is None:
        p_m = np.array([1 / len(M)] * len(M))
        # p_m = p_m.reshape((8, 40))
        p_m = p_m.reshape(grid.shape)   

    cost_ = 0

    for c in C:
        coords = np.argwhere(grid == c)
        for m in M:
            # coordiantes should be mapped using cielab_dict
            coords_cielab = np.array([cielab_dict[tuple(k)] for k in coords.tolist()])
            distance = np.linalg.norm(cielab_dict[m] - coords_cielab) ** 2
            cost_ += np.exp( - (p_m[m] * distance))
    return cost_


def informativeness(grid, p_m, cielab_dict):
    '''
    Compute the informativeness of a grid using the cost function from Carr et al. (2020).

    Parameters:
        grid (numpy.ndarray): The grid to compute the informativeness of.
        p_m (numpy.ndarray): The prior over each cell in the grid.
        cielab_dict (dict): A dictionary mapping the coordinates of each cell to its CIELAB color.

    Returns:
        float: The informativeness of the grid.
    '''
    # return safelog(2 ** (-cost(grid, p_m)))
    return -safelog(cost(grid, p_m, cielab_dict))


def point_in_hull(point, hull):
    '''
    Check if a point is inside the convex hull.
    '''
    return all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations)

def degree_of_convexity_cluster(cluster_points):
    '''
    Build the convex hull around cluster. 
    Find the number of points in the hull. 
    Divide the number of points in the class 
    by the number of points inside the hull
    '''
    if len(cluster_points) < 4:
        return 1
    else:
        try:
            hull = ConvexHull(cluster_points)
            in_hull = [p for p in cluster_points if point_in_hull(p, hull)]
            return len(in_hull) / (len(cluster_points) + 1)
        except:
            return 0 

def degree_of_convexity(arrays):
    '''Compute the degree of convexity for the given array or arrays'''
    convexity_list = []
    for arr in arrays:
        labels = np.unique(arr)  # Get unique labels
        convexities = [degree_of_convexity_cluster(np.argwhere(arr == label)) for label in labels]
        weights = [np.sum(arr == label) for label in labels]
        convexity = np.average(convexities, weights=weights)
        convexity_list.append(convexity)
    return np.array(convexity_list)