import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import f1_score


def normalize_labels(arr):
    '''Reorder the labels in an array to be consecutive integers starting from 0'''
    # Find the unique labels in the array
    unique_labels = np.unique(arr)
    # Create a mapping from the old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    # Apply the mapping to the array
    normalized_array = np.vectorize(label_mapping.get)(arr)
    return normalized_array


def f1_macro(matrix1, matrix2):
    '''Compute the macro F1 score between two color grids'''
    # Flatten the matrices
    labels1 = matrix1.flatten()
    labels2 = matrix2.flatten()
    
    # Calculate the macro F1 score directly
    macro_f1 = f1_score(labels1, labels2, average='weighted')
    
    return macro_f1

def are_points_coplanar(points):
    '''Check if a set of points are coplanar, or collinear in 2D'''
    if len(points) < 3:
        return True  # cannot build a convex hull with less than 3 points

    # Determine the dimensionality of the points
    dim = len(points[0])
    
    if dim == 2:
        # Use the first point as the reference
        p0 = np.array(points[0])
        vectors = [np.array(p) - p0 for p in points[1:]]
        # Check if all vectors are multiples of each other (indicating collinearity)
        for i in range(1, len(vectors)):
            if not np.allclose(np.cross(vectors[0], vectors[i]), 0):
                return False
        return True

    else:
        # Use the first point as the reference
        p0 = np.array(points[0])
        vectors = [np.array(p) - p0 for p in points[1:]]

        # Create a matrix of vectors
        matrix = np.vstack(vectors)

        # Check if the rank of the matrix is less than the dimensionality (indicating coplanarity)
        rank = np.linalg.matrix_rank(matrix)
        return rank < dim

def safelog(x):
    '''
    Compute the logarithm of a positive number x, or return 0 if x is non-positive.
    '''
    return np.log(x) if x > 0 else 0


def non_dominated_front(points_x, points_y):
    """
    Returns the non-dominated (Pareto) front of a list of 2-D points using Kung's algorithm.
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
    """
    return np.log(len(gzip.compress(grid.tobytes(), compresslevel=9)))


def cost(grid, p_m, cielab_dict):
    '''
    Cost function from Carr et al. (2020) for the informativeness of a grid.
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
    '''
    return -safelog(cost(grid, p_m, cielab_dict))


def point_in_hull(point, hull, tolerance=1e-12):
    '''
    Check if a point is inside the convex hull.
    '''
    return np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= tolerance)


def degree_of_convexity_cluster(cluster_points, all_coords):
    '''
    Estim§ate the degree of convexity for a cluster of points.
    '''
    if not are_points_coplanar(cluster_points):
        hull = ConvexHull(cluster_points)
        in_hull = [p for p in all_coords if point_in_hull(p, hull)]
        return len(cluster_points) / len(in_hull) 
    else:
        return 1


def degree_of_convexity(arrays, all_coords=None):
    '''
    Compute the degree of convexity for the given array or arrays
    '''
    if all_coords is None:
        # Get all coordinates in the grid
        all_coords = [(i, j) for i in range(arrays[0].shape[0]) 
                             for j in range(arrays[0].shape[1])]
    convexity_list = []
    for arr in arrays:
        labels = np.unique(arr)  # Get unique labels
        convexities = [degree_of_convexity_cluster(np.argwhere(arr == label), all_coords) for label in labels]
        weights = [np.sum(arr == label) for label in labels]
        convexity = np.average(convexities, weights=weights)
        convexity_list.append(convexity)
    return np.array(convexity_list)


def degree_of_convexity_cielab(arrays, coord_dict, all_coords):
    '''Compute the degree of convexity for the given array or arrays, but get coordinates from a dictionary (coordinate on a grid --> CIELAB coordinates)'''
    convexity_list = []
    for arr in arrays:
        labels = np.unique(arr)  # Get unique labels
        convexities = [degree_of_convexity_cluster([coord_dict[tuple(coord)] for coord 
                                                    in np.argwhere(arr == label)], all_coords) for label in labels]
        weights = [np.sum(arr == label) for label in labels]
        convexity = np.average(convexities, weights=weights)
        convexity_list.append(convexity)
    return np.array(convexity_list)


def plot_color_grid(grid, rgb_dict, prior_m_matrix):
    """
    Plots a grid with colors corresponding to categories and their respective RGB values.
    """
    
    # Create an empty dictionary to store the RGB values for each unique category in the grid
    color_map = dict()

    # Loop through each unique category in the grid
    for color_c in np.unique(grid):
        # Get the coordinates of all cells belonging to the current category
        coordinates = np.argwhere(grid == color_c)
        
        # Extract probabilities for each coordinate from prior_m_matrix and normalize them
        probs = [prior_m_matrix[coord[0], coord[1]] for coord in coordinates]
        probs = np.array(probs) / sum(probs)

        # Find the coordinate with the highest probability
        max_prob_coord = coordinates[np.argmax(probs)]
        
        # Map the category to its corresponding RGB value
        color_map[color_c] = rgb_dict[max_prob_coord[0] + 1, max_prob_coord[1] + 1]

    # Create an empty image with the same shape as the grid but with an additional dimension for RGB channels
    image = np.zeros((grid.shape[0], grid.shape[1], 3))

    # Assign colors to each cell in the image based on the grid values and color_map
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            image[i, j] = color_map[grid[i, j]]

    return image


def plot_n_neighbours(idx_list, n, fig_name, lg_color, prior_m_matrix, 
                      cielab_dict, rgb_dict, lang_info, mdl_values, cost_values, lgs):
    '''
    Plot the n-nearest neighbors for each language given simplicity and informativeness values.
    '''
    k = len(idx_list)  # Number of indices in the list
    fig, ax = plt.subplots(n, k, figsize=(8, 3))  # Adjust figsize as needed

    for col, idx in enumerate(idx_list):
        distances = []
        comp = compression(lg_color[idx])
        info = informativeness(lg_color[idx], prior_m_matrix, cielab_dict)

        for i, j in zip(mdl_values, cost_values):
            distances.append(np.sqrt((i - comp)**2 + (j - info)**2))

        # Sort distances (from smallest to largest) and get the indices
        nearest_indices = np.argsort(distances)[:n]

        for row, nearest_idx in enumerate(nearest_indices):
            if row == 0:
                ax[row, col].set_title(lang_info[idx])
                img = plot_color_grid(lg_color[idx], rgb_dict, prior_m_matrix)
                ax[row, col].imshow(img, interpolation='none')
            else:
                ax[row, col].set_title(f'{row}-st nearest', fontstyle='italic')
                img = plot_color_grid(lgs[nearest_idx], rgb_dict, prior_m_matrix)
                ax[row, col].imshow(img, interpolation='none')
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # remove border
            for spine in ax[row, col].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'figures/{fig_name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def letter_subplots(axes=None, letters=None, xoffset=-0.1, yoffset=1.0, **kwargs):
    """Add letters to the corners of subplots (panels). By default each axis is
    given an uppercase bold letter label placed in the upper-left corner.
    Args
        axes : list of pyplot ax objects. default plt.gcf().axes.
        letters : list of strings to use as labels, default ["A", "B", "C", ...]
        xoffset, yoffset : positions of each label relative to plot frame
          (default -0.1,1.0 = upper left margin). Can also be a list of
          offsets, in which case it should be the same length as the number of
          axes.
        Other keyword arguments will be passed to annotate() when panel letters
        are added.
    Returns:
        list of strings for each label added to the axes
    Examples:
        Defaults:
            >>> fig, axes = plt.subplots(1,3)
            >>> letter_subplots() # boldfaced A, B, C
        
        Common labeling schemes inferred from the first letter:
            >>> fig, axes = plt.subplots(1,4)        
            >>> letter_subplots(letters='(a)') # panels labeled (a), (b), (c), (d)
        Fully custom lettering:
            >>> fig, axes = plt.subplots(2,1)
            >>> letter_subplots(axes, letters=['(a.1)', '(b.2)'], fontweight='normal')
        Per-axis offsets:
            >>> fig, axes = plt.subplots(1,2)
            >>> letter_subplots(axes, xoffset=[-0.1, -0.15])
            
        Matrix of axes:
            >>> fig, axes = plt.subplots(2,2, sharex=True, sharey=True)
            >>> letter_subplots(fig.axes) # fig.axes is a list when axes is a 2x2 matrix

    See also: https://github.com/matplotlib/matplotlib/issues/20182
    """

    # get axes:
    if axes is None:
        axes = plt.gcf().axes
    # handle single axes:
    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # set up letter defaults (and corresponding fontweight):
    fontweight = "bold"
    ulets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)])
    llets = list('abcdefghijklmnopqrstuvwxyz'[:len(axes)])
    if letters is None or letters == "A":
        letters = ulets
    elif letters == "(a)":
        letters = [ "({})".format(lett) for lett in llets ]
        fontweight = "normal"
    elif letters == "(A)":
        letters = [ "({})".format(lett) for lett in ulets ]
        fontweight = "normal"
    elif letters in ("lower", "lowercase", "a"):
        letters = llets

    # make sure there are x and y offsets for each ax in axes:
    if isinstance(xoffset, (int, float)):
        xoffset = [xoffset]*len(axes)
    else:
        assert len(xoffset) == len(axes)
    if isinstance(yoffset, (int, float)):
        yoffset = [yoffset]*len(axes)
    else:
        assert len(yoffset) == len(axes)

    # defaults for annotate (kwargs is second so it can overwrite these defaults):
    my_defaults = dict(fontweight=fontweight, fontsize='large', ha="center",
                       va='center', xycoords='axes fraction', annotation_clip=False)
    kwargs = dict( list(my_defaults.items()) + list(kwargs.items()))

    list_txts = []
    for ax,lbl,xoff,yoff in zip(axes,letters,xoffset,yoffset):
        t = ax.annotate(lbl, xy=(xoff,yoff), **kwargs)
        list_txts.append(t)
    return list_txts
