import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial import distance
from src.colorconv import ConvexCluster
import pandas as pd
from tqdm import trange

# Helper function to generate lexicons
def generate_lexicons(N, shape, n, max_attempts=10000, 
                      coords=None, coords_dict=None):
    LS = []
    convexities = []
    attempts = 0

    while len(LS) < n and attempts < max_attempts:
        lexicons = ConvexCluster(
            X=np.zeros(shape),
            d=0.1,
            N=N,
            s=np.random.uniform(0, 1),
            c=np.random.uniform(0, 1),
            coords=coords,
            reverse_labels=coords_dict
        )
        lex = lexicons.run()

        if len(np.unique(lex)) == N:
            LS.append(lex)
            convexities.append(lexicons.degree_of_convexity())
        attempts += 1

    if len(LS) < n:
        lexs, convx = generate_lexicons(N, shape, n - len(LS), max_attempts, coords, coords_dict)
        LS.extend(lexs)
        convexities.extend(convx)

    return np.array(LS).astype(int), np.array(convexities)

# Helper function to calculate normalized mutual information
def calculate_normalized_mutual_info(data, n_iter, n_rounds):
    mis = []

    for k in range(n_iter):
        mi = []
        for i in range(n_rounds):
            mi.append(normalized_mutual_info_score(
                data.logs[k]['a1'][i].ravel(), 
                data.logs[k]['a2'][i].ravel()
            ))
        mis.append(mi)

    return pd.DataFrame(mis).melt()

# Base class for RSA model
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
        cost_arrays = []
        unique_labels = np.unique(L)
        for subarray in L:
            cost_p = np.zeros(max(unique_labels) + 1)
            for c in unique_labels:
                coords = np.argwhere(subarray == c)
                for m in coords:
                    for n in coords:
                        cost_p[c] += distance.euclidean(m, n)
                cost_p[c] /= (len(coords) * len(coords))
            cost_arrays.append(cost_p)
        return np.nan_to_num(np.expand_dims(np.array(cost_arrays), axis=1))

    def L_0(self, L):
        '''Literal listener'''
        return self.normalize(self.binary_language(L) * self.prior)
    
    def S_p(self, L):
        '''Pragmatic speaker without cost'''
        return self.normalize(np.exp(self.alpha * self.safelog(self.L_0(L).transpose(0, 2, 1))))
    
    def L_p(self, L):
        '''Pragmatic listener'''
        return self.normalize(self.S_p(L).transpose(0, 2, 1) * self.prior)

# Agent class inheriting from BaseRSA
class Agent(BaseRSA):
    def __init__(self, alpha, prior, shape, convex, Lexicons, 
                 convexities, threshold, beta, simple):
        super().__init__(alpha, prior)
        self.n_meanings = shape[0] * shape[1]
        self.Lexicons = Lexicons
        self.convexities = convexities
        self.convex = convex
        self.beta = beta
        self.simple = simple
        
        self.Lexicons = self.Lexicons[self.convexities > threshold]
        self.convexities = self.convexities[self.convexities > threshold]
        self.convexities /= np.sum(self.convexities)

        if simple:
            self.simplicity = [1 / compression(subarray) for subarray in self.Lexicons]
            self.simplicity /= np.sum(self.simplicity)
        
        self.prob_lexicon = np.ones(len(self.Lexicons)) / len(self.Lexicons)
        self.precompute_S_p()
        self.precompute_L_p()
        self.n_words = np.max([np.max(subarray) for subarray in self.Lexicons]) + 1
    
    def precompute_S_p(self):
        self.S_p_values = self.S_p(self.Lexicons)
    
    def precompute_L_p(self):
        self.L_p_values = self.L_p(self.Lexicons)
    
    def speaker(self, m):
        lexicon_idx = np.random.choice(np.arange(len(self.Lexicons)), p=self.prob_lexicon)
        if np.sum(self.S_p_values[lexicon_idx][m]) == 0:
            return np.random.choice(np.arange(self.n_words), p= [1 / self.n_words] * self.n_words)
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
            self.prob_lexicon = self.normalize(self.convexities ** (self.beta) * self.prob_lexicon)
        if self.simple:
            self.prob_lexicon = self.normalize(self.simplicity ** (self.beta) * self.prob_lexicon)
        if role == "speaker" and correct:
            self.prob_lexicon = self.normalize(self.S_p_values[:, m, w] * self.prob_lexicon)
        elif role == "listener" and correct:
            self.prob_lexicon = self.normalize(self.L_p_values[:, w, m] * self.prob_lexicon)

# Experiment class
class Experiment:
    def __init__(self, alpha, prior, shape, n, n_iter, n_rounds, 
                 Lexicons, convex, threshold, beta, simple, convexities):
        self.n_iter = n_iter
        self.n_rounds = n_rounds
        self.logs = {}
        self.alpha = alpha
        self.prior = prior
        self.convex = convex
        self.simple = simple
        self.threshold = threshold
        self.beta = beta
        self.shape = shape
        self.n = min(n, 300000)
        self.Lexicons = Lexicons
        self.convexities = convexities

    def sample_meaning(self):
        return np.random.choice(range(self.shape[0] * self.shape[1]))

    def one_round(self, a, b, m):
        w = a.speaker(m)
        g = b.listener(w)
        a.update(w, m, m == g, "speaker")
        b.update(w, m, m == g, "listener")
    
    def run(self):
        for i in range(self.n_iter):
            p1 = np.random.permutation(len(self.Lexicons))
            p2 = np.random.permutation(len(self.Lexicons))
            agents = [
                Agent(self.alpha, self.prior, self.shape, self.convex, self.Lexicons[p1], self.convexities[p1], self.threshold, self.beta, self.simple),
                Agent(self.alpha, self.prior, self.shape, self.convex, self.Lexicons[p2], self.convexities[p2], self.threshold, self.beta, self.simple)
            ]
            lexicons_ = {'a1': [], 'a2': []}
            for r in range(self.n_rounds):
                lexicons_['a1'].append(agents[0].Lexicons[np.argmax(agents[0].prob_lexicon)])
                lexicons_['a2'].append(agents[1].Lexicons[np.argmax(agents[1].prob_lexicon)])
                m = self.sample_meaning()
                if r % 2 == 0:
                    self.one_round(agents[0], agents[1], m)
                else:
                    self.one_round(agents[1], agents[0], m)
            self.logs[i] = lexicons_

# Plot function for lexicons
def plot_last_lexicons(model, n_iter, pos=-1, 
                       n_examples=5, title=None):
    # Extract lexicon pairs
    lexicon_pairs = [(tuple(model.logs[i]['a1'][pos].flatten()), tuple(model.logs[i]['a2'][pos].flatten())) for i in range(n_iter)]
    
    # Count frequency of each pair
    lexicon_counter = Counter(lexicon_pairs)

    # print the number of distinct lexicon pairs
    print(f"Number of distinct lexicon pairs: {len(lexicon_counter)}")
    
    # Get the most frequent lexicon pairs
    most_frequent_pairs = lexicon_counter.most_common(n_examples)
    
    # Plot the most frequent lexicons
    fig, ax = plt.subplots(2, n_examples, figsize=(5, 2))  # Adjusted figsize for better visibility
    
    for i, ((a1, a2), freq) in enumerate(most_frequent_pairs):
        ax[0, i].imshow(np.array(a1).reshape(model.logs[0]['a1'][pos].shape), cmap='tab10', interpolation='none')
        ax[1, i].imshow(np.array(a2).reshape(model.logs[0]['a2'][pos].shape), cmap='tab10', interpolation='none')
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        total = sum(lexicon_counter.values())
        ax[0, i].set_title(f"{round((freq / total) * 100, 2)}%")
    
    ax[0, 0].set_ylabel('A1')
    ax[1, 0].set_ylabel('A2')
    
    if title:
        # save to pdf
        plt.savefig(f'{title}.pdf', bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    plt.show()

def generate_coords(shape):
    coords = np.array(np.meshgrid(np.arange(shape[0]), 
                                  np.arange(shape[1]))).T.reshape(-1, 2)
    coords = [tuple(i) for i in coords]

    # dictionary mapping coords to themselves
    coords_dict = {coord: coord for coord in coords}
    
    return coords, coords_dict


def efficiency_simplicity_analysis(n, shape, N_min, N_max, 
                                   beta, n_iter=500, n_rounds=100,
                                   coords=None, coords_dict=None):

    LS_ = []
    convex_lexicons = []
    non_convex_lexicons = []
    simplicity_lexicons = []
    # simplicity_convex_lexicons = []

    # mi_convex = []
    # mi_nconvex = []
    # mi_simple = []

    coords, coords_dict = generate_coords(shape)

    for N in trange(N_min, N_max):

        LS, cx = generate_lexicons(N, shape, n, coords=coords, coords_dict=coords_dict)

        # prior = np.ones(shape[1] * shape[0]) / shape[1] * shape[0]
        
        a = Experiment(alpha=1, 
               shape=shape, 
               n=len(LS), 
               n_iter=n_iter, 
               n_rounds=n_rounds,
               prior= np.ones(shape[1] * shape[0]) / shape[1] * shape[0], 
               convex=True,
               threshold=0,
               beta=beta,
               Lexicons=LS,
               simple=False,
               convexities=cx)
        a.run()

        convex_lexicons.append([a.logs[i]['a1'][-1] for i in range(n_iter)])
        convex_lexicons.append([a.logs[i]['a2'][-1] for i in range(n_iter)])

        # calculate normalized mutual information for last round using list comprehension
        mi_convex = [normalized_mutual_info_score(a.logs[i]['a1'][-1].ravel(), 
                                                       a.logs[i]['a2'][-1].ravel()) for i in range(n_iter)]

        b = Experiment(alpha=1,
                shape=shape,
                n=len(LS),
                n_iter=n_iter,
                n_rounds=n_rounds,
                prior= np.ones(shape[1] * shape[0]) / shape[1] * shape[0],
                convex=False,
                threshold=0,
                beta=beta,
                Lexicons=LS,
                simple=False,
                convexities=cx)
        b.run()
        
        non_convex_lexicons.append([b.logs[i]['a1'][-1] for i in range(n_iter)])
        non_convex_lexicons.append([b.logs[i]['a2'][-1] for i in range(n_iter)])

        # calculate normalized mutual information for last round using list comprehension
        mi_nconvex = [normalized_mutual_info_score(b.logs[i]['a1'][-1].ravel(),
                                                        b.logs[i]['a2'][-1].ravel()) for i in range(n_iter)]

        c = Experiment(alpha=1,
                    shape=shape,
                    n=len(LS),
                    n_iter=n_iter,
                    n_rounds=n_rounds,
                    prior= np.ones(shape[1] * shape[0]) / shape[1] * shape[0],
                    convex=False,
                    threshold=0,
                    beta=beta,
                    Lexicons=LS,
                    simple=True,
                    convexities=cx)
        c.run()

        simplicity_lexicons.append([c.logs[i]['a1'][-1] for i in range(n_iter)])
        simplicity_lexicons.append([c.logs[i]['a2'][-1] for i in range(n_iter)])

        # calculate normalized mutual information for last round using list comprehension
        mi_simple = [normalized_mutual_info_score(c.logs[i]['a1'][-1].ravel(),
                                                      c.logs[i]['a2'][-1].ravel()) for i in range(n_iter)]
        
        # d = Experiment(alpha=1,
        #             shape=shape,
        #             n=len(LS),
        #             n_iter=n_iter,
        #             n_rounds=n_rounds,
        #             prior= np.ones(shape[1] * shape[0]) / shape[1] * shape[0],
        #             convex=False,
        #             treshold=0.95,
        #             # beta=beta,
        #             beta = 1,
        #             Lexicons=LS,
        #             simple=True)
        # d.run()

        # simplicity_convex_lexicons.append([d.logs[i]['a1'][-1] for i in range(n_iter)])
        # simplicity_convex_lexicons.append([d.logs[i]['a2'][-1] for i in range(n_iter)])

        # # calculate normalized mutual information for last round using list comprehension
        # mi_simple_conv = [normalized_mutual_info_score(d.logs[i]['a1'][-1].ravel(),
        #                                               d.logs[i]['a2'][-1].ravel()) for i in range(n_iter)]

        LS_.append(LS)

    # # flatten the list of lists lexicons
    # lexicons_convex = [item for sublist in lexicons_convex for item in sublist]
    # lexicons_nconvex = [item for sublist in lexicons_nconvex for item in sublist]
    # flatten the list of lists lexicons
    convex_lexicons = [item for sublist in convex_lexicons for item in sublist]
    non_convex_lexicons = [item for sublist in non_convex_lexicons for item in sublist]
    simplicity_lexicons = [item for sublist in simplicity_lexicons for item in sublist]
    # simplicity_convex_lexicons = [item for sublist in simplicity_convex_lexicons for item in sublist]

    # flatten LS_
    LS_ = [item for sublist in LS_ for item in sublist]
    
    # return convex_lexicons, non_convex_lexicons, simplicity_lexicons, LS_, coords, coords_dict, mi_convex, mi_nconvex, mi_simple
    return convex_lexicons, non_convex_lexicons, simplicity_lexicons, LS_, coords, coords_dict, mi_convex, mi_nconvex, mi_simple
        