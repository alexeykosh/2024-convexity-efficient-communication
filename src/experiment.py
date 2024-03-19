import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import distance

NOISE = 0


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


class BaseRSA:
    def __init__(self, alpha, prior):
        self.alpha = alpha
        self.prior = prior

    @staticmethod
    def safelog(vals):
        with np.errstate(divide='ignore'):
            return np.log(vals)
        
    @staticmethod
    def binary_language(L):
        max_labels = np.max([np.max(subarray) for subarray in L]) + 1
        binary_arrays = []
        for subarray in L:
            x = max_labels
            y = np.prod(subarray.shape)
            binary_array = np.zeros((x, y), dtype=np.uint8)
            for i, row in enumerate(subarray):
                for j, val in enumerate(row):
                    binary_array[val, i * subarray.shape[1] + j] = 1
            binary_arrays.append(binary_array)
        return np.array(binary_arrays)
    
    @staticmethod
    def normalize(vals):
        return np.nan_to_num(vals / np.sum(vals, axis=-1, keepdims=True))
    
    @staticmethod
    def cost(L):
        # for each unique label, compute pairwise distance between all points
        # and sum them up
        cost_arrays = []
        unique_labels = np.unique(L)
        for subarray in L:
            cost_p = np.zeros(max(unique_labels) + 1)
            for c in unique_labels:
                coords = np.argwhere(subarray == c)
                for m in coords:
                    for n in coords:
                        cost_p[c] += distance.euclidean(m, n)
                cost_p[c] = cost_p[c] / (len(coords) * (len(coords)))
            cost_arrays.append(cost_p)
        return np.nan_to_num(np.expand_dims(np.array(cost_arrays), axis=1))

    def L_0(self, L):
        '''Literal listener'''
        return self.normalize(self.binary_language(L) * self.prior )  
        # *  self.informativeness(L)
    
    def S_p(self, L):
        return self.normalize(np.exp(self.alpha * (self.safelog(self.L_0(L).transpose(0, 2, 1)))))
        # return self.normalize(np.exp(self.alpha * (self.safelog(self.L_0(L).transpose(0, 2, 1)) - self.cost(L))))
    
    def L_p(self, L):
        return self.normalize(self.S_p(L).transpose(0, 2, 1) * self.prior)
        # * self.informativeness(L)


class Agent(BaseRSA):
    def __init__(self, alpha, prior, N, shape, convex, Lexicons, convexities, treshold, beta):
        super().__init__(alpha, prior)
        self.n_words = N
        self.n_meanings = shape[0] * shape[1]

        self.Lexicons = Lexicons
        self.convexities = convexities

        self.convex = convex
        self.beta = beta
        
        self.Lexicons = self.Lexicons[self.convexities > treshold]
        self.convexities = self.convexities[self.convexities > treshold]
        
        self.prob_lexicon = np.ones(len(self.Lexicons)) / len(self.Lexicons)

        self.precompute_S_p()
        self.precompute_L_p()
    
    def precompute_S_p(self):
        self.S_p_values = self.S_p(self.Lexicons)
    
    def precompute_L_p(self):
        self.L_p_values = self.L_p(self.Lexicons)
    
    def speaker(self, m):
        lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
        if np.sum(self.S_p_values[lexicon_idx][m]) == 0:
            return np.random.choice(np.arange(self.n_words), p=[1 / self.n_words] * self.n_words)
        else:
            return np.random.choice(np.arange(self.n_words), p=self.S_p_values[lexicon_idx][m])
    
    def listener(self, w):
        lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
        if np.sum(self.L_p_values[lexicon_idx][w]) == 0:
            return np.random.choice(np.arange(self.n_meanings), p=[1 / self.n_meanings] * self.n_meanings)
        else:
            return np.random.choice(np.arange(self.n_meanings), p=self.L_p_values[lexicon_idx][w])

    def update(self, w, m, correct, role):
        if self.convex:
            self.prob_lexicon = self.normalize(self.convexities**(1/self.beta) * self.prob_lexicon)
        if role == "speaker":
            if correct:
                self.prob_lexicon = self.normalize(self.S_p_values[:, m, w] \
                                                    * self.prob_lexicon + NOISE)
        elif role == "listener":
            if correct:
                self.prob_lexicon = self.normalize(self.L_p_values[:, w, m] \
                                                        * self.prob_lexicon  + NOISE)


class Experiment:
    def __init__(self, alpha, prior, shape, n, n_iter, n_rounds, N, convex, treshold, beta):
        self.n_iter = n_iter
        self.n_rounds = n_rounds
        
        self.lexicons = {}

        self.alpha = alpha
        self.prior = prior
        self.convex = convex

        self.treshold = treshold
        self.beta = beta

        self.shape = shape
        if n > 300000:
            self.n = 300000
        else:
            self.n = n 
            self.treshold = 0
        self.N = N


    def sample_meaning(self):
        return np.random.choice([i for i in range(self.shape[0] * self.shape[1])])

    def one_round(self, a, b, m, i, r):
        w = a.speaker(m)
        g = b.listener(w)

        a.update(w, m, m == g, "speaker")
        b.update(w, m, m == g, "listener")

        self.logs[i][r]['word'] = w
        self.logs[i][r]['guess'] = g
        self.logs[i][r]['correct'] = 1 if (m == g) else 0
    
    def run(self):
        
        # Lexicons = generate_lexicons(N=self.N, shape=self.shape, n=self.n)
        ####### !!!!!!!! #######
        # NB: computing on lexicons from ColorConvex 
        Lexicons = LS 
        ####### !!!!!!!! #######
        convexities = degree_of_convexity(Lexicons)
        
        for i in range(self.n_iter):
            p1 = np.random.permutation(len(Lexicons))
            p2 = np.random.permutation(len(Lexicons))
            agents = [Agent(alpha=self.alpha, prior=self.prior, N=self.N, 
                            shape=self.shape, convex=self.convex, 
                            Lexicons= Lexicons[p1], convexities=convexities[p1], 
                            treshold=self.treshold, beta=self.beta),
                      Agent(alpha=self.alpha, prior=self.prior, N=self.N, 
                            shape=self.shape, convex=self.convex, 
                            Lexicons=Lexicons[p2], convexities=convexities[p2], 
                            treshold=self.treshold, beta=self.beta)]

            lexicons_ = {'a1': [], 'a2': []}
            for r in range(self.n_rounds):

                lexicons_['a1'].append(agents[0].Lexicons[np.argmax(agents[0].prob_lexicon)])
                lexicons_['a2'].append(agents[1].Lexicons[np.argmax(agents[1].prob_lexicon)])
                
                m = self.sample_meaning()
                
                self.logs[i][r]['meaning'] = m
                if r % 2 == 0:
                    self.one_round(agents[0], 
                                   agents[1], 
                                   m, i, r)
                else:
                    self.one_round(agents[1], 
                                   agents[0], 
                                    m, i, r)
                

            self.lexicons[i] = lexicons_
