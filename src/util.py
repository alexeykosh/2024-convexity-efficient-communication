from numcompress import compress
import numpy as np

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

def safelog(vals):
    with np.errstate(divide='ignore'):
        return np.log(vals)

def compression(grid):
    return len(compress(grid.flatten().tolist()))


def cost(grid, p_m, cielab_dict):
    # get all unique values in the grid
    C = np.unique(grid)
    # get list of all coordinates of each cell in the grid
    M = [(i, j) for i in range(grid.shape[0]) 
                       for j in range(grid.shape[1])]
    # p_m equals to 1 divided by the number of cells in the grid
    if p_m is None:
        p_m = np.array([1 / len(M)] * len(M))
        p_m = p_m.reshape((8, 40))   

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
    # return safelog(2 ** (-cost(grid, p_m)))
    return -cost(grid, p_m, cielab_dict)