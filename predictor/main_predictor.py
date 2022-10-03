import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from scipy.special import gamma
from collections import defaultdict
from itertools import product, combinations
import pickle
import networkx as nx
from tqdm import tqdm




class Wishart:
    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level

    def fit(self, X):
        from sklearn.neighbors import KDTree
        kdt = KDTree(X, metric='euclidean')

        distances, neighbors = kdt.query(X, k = self.wishart_neighbors + 1, return_distance = True)
        neighbors = neighbors[:, 1:]


        distances = distances[:, -1]
        indexes = np.argsort(distances)
        
        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype = int) - 1

        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)

        for index in indexes:
            neighbors_clusters =\
                np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]


            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level

                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue

                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                self.clusters_to_objects[not_sig_cluster].clear()
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue

                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                self.clusters_to_objects[cur_cluster].clear()

                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)

        return self.clean_data()

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq :  index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype = int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis = 0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)





def self_healing_window(train_array, heal_array, not_predictable_point_label = 'N'):
    healed_array = heal_array
    nnp_label = not_predictable_point_label

    n = len(train_array)
    k = len(heal_array)

    checked_windows = []

    for i in range(n - k):
        error = 0
        
        for j in range(k):
            if heal_array[j] != nnp_label:
                error += (train_array[i + j] - heal_array[j]) ** 2
        
        checked_windows.append([i, error])
    
    best_window = min(checked_windows, key = lambda x: x[1])

    for i in range(k):
        if heal_array[i] == nnp_label:
            heal_array[i] = train_array[i + best_window[0]]

    return healed_array



class predictor_markov_chain():
    '''
    Class for time series predict
    Algorithm to start:
       -> Create a class
       -> Fit train array with fit_train_array
       -> Produce a graph with produce_graph
       -> Produce a predict with predict
    Attributes:
    ---------------------------
    graph : networkx graph
        Trained and used for predictions
    train_array : list of size len(list)
        List or array is used to train graph
    max_pattern_len : int
        Max possible length for array which is produced from predictor
    '''

    def __init__(self):
        self.graph = None
        self.train_array = None
        self.max_pattern_len = None

    def fit_train_array(self, train_array):
        self.train_array = train_array

    def produce_graph(self, max_pattern_len = 100, round_ = 1, mode = 'ind'):

        if self.train_array.all() == None:
            raise ValueError('Train array is empty')

        if len(self.train_array) < max_pattern_len:
            raise ValueError('Max pattern length is greater than train array length')

        G = nx.DiGraph()
        n = len(self.train_array)
        G.add_node(0, diff = None, level = 0)

        for i in tqdm(range(n - max_pattern_len)):
            current_node = 0

            for j in range(max_pattern_len - 1):
                diff = round(self.train_array[i + j] - self.train_array[i + j + 1], round_)

                if mode == 'uni':
                    current_level = [i for i in G.nodes if G.nodes[i]['level'] == j + 1]
                elif mode == 'ind':
                    current_level = [i for i in G.neighbors(current_node)]

                if diff not in [G.nodes[i]['diff'] for i in current_level]:
                    new_node = len(G)
                    G.add_node(new_node, diff = diff, level = j + 1)
                    G.add_edge(current_node, new_node, weight = 1)
                    current_node  = new_node
                
                else:
                    correct_node = [i for i in current_level if G.nodes[i]['diff'] == diff][0]
                    
                    if G.has_edge(current_node, correct_node):
                        G[current_node][correct_node]['weight'] += 1
                    
                    else:
                        G.add_edge(current_node, correct_node, weight = 1)

                    if G.nodes[correct_node]['level'] - G.nodes[current_node]['level'] != 1:
                        raise IndexError()
                        
                    current_node = correct_node

    
        self.graph = G
        self.max_pattern_len = max_pattern_len
        pass

    def get_graph(self):
        return self.graph
    
    def get_max_pattern_len(self):
        return self.max_pattern_len

    def set_graph(self, graph, max_pattern_len):
        self.graph = graph
        self.max_pattern_len = max_pattern_len
        pass
    
    def get_trajectory(self,start_y, trajectory_len, power = 1):

        if self.graph == None:
            raise ValueError('Graph not founded')

        if trajectory_len > self.max_pattern_len:
            raise ValueError('Trajectory length > Max possible trajectory lenght')

        current_node = 0
        current_y = start_y
        trajectory = [start_y]

        for i in range(trajectory_len - 2):

            weights = np.array([self.graph[current_node][i]['weight'] for i in self.graph.neighbors(current_node)]) ** power
            nodes = [i for i in self.graph[current_node]]
            weights = weights / weights.sum()

            if np.array(weights).sum() == 0:
                return trajectory

            next_node = np.random.choice(nodes, p = weights)
            current_y = current_y - self.graph.nodes[next_node]['diff']
            trajectory.append(current_y)
            current_node = next_node
            
        return trajectory

    def continue_trajectory(self, trace, power = 1):
        trajectory = [trace[0]]
        current_node = 0

        for i in range(len(trace) - 1):
            diff = trace[i] - trace[i + 1]
            nodes = [(i, self.graph.nodes[i]['diff']) for i in self.graph.neighbors(current_node)]
            error = [(i, np.sqrt((j - diff) ** 2)) for i, j in nodes]
            next_node = min(error, key = lambda x: x[1])[0]

            trajectory.append(trajectory[i] - self.graph.nodes[next_node]['diff'])
            current_node = next_node
        
        current_y = trajectory[-1]

        for i in range(self.max_pattern_len - len(trace)):
            
             weights = np.array([self.graph[current_node][i]['weight'] for i in self.graph.neighbors(current_node)]) ** power
             nodes = [i for i in self.graph[current_node]]
             weights = weights / weights.sum()

             next_node = np.random.choice(nodes, p = weights)
             current_y = current_y - self.graph.nodes[next_node]['diff']
             trajectory.append(current_y)
             current_node = next_node
            
        return trajectory

    def unified_prediction(self, possible_predictions, up_method, \
        random_perturbation=False, **kwargs):
        '''Calculates unified prediciton from set of possible predicted values

        Parameters
        ----------
        possible_predictions: list or 1D np.array
            List of possible predictions
        up_method: str from {'a', 'wi', 'db', 'op'}
            Method of estimating unified prediciton
            'a'  - average
            'wi' - clustering with Wishart, get largest cluster mean
            'db' - clustering with DBSCAN, get largest cluster mean
            'op' - clustering with OPTICS, get largest cluster mean
        random_perturbation: boolean
            Add noise to unified prediction

        **kwargs
        -- for Wishart, DBSCAN and OPTICS clustering --
        min_samples: int > 1 or float between 0 and 1, default=5
            Minimal number of samples in cluster
        eps: float from 0 to 1, default=0.01
            Max distance within one cluster
        cluster_size_threshold: float from 0 to 1, default=0.2
            Minimal percentage of points in largest cluster to call point predictable
        one_cluster_rule: boolean, defalut=False
            Point is predictable only is there is one cluster (not including noise)
        '''

        if len(possible_predictions) == 0:
            return 'N'
        min_samples = kwargs.get('min_samples', 5)
        eps = kwargs.get('eps', 0.01)

        if up_method == 'a':
            avg = np.mean(possible_predictions)
            if random_perturbation:
                avg += np.random.normal(0, 0.01)
            return avg

        if up_method == 'wi' or up_method == 'db' or up_method=='op':  
            try: 
                if up_method == 'db': 
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
                    
                elif up_method == 'wi':
                    clustering = Wishart(min_samples, eps)
                    labels = clustering.fit(np.array(possible_predictions).reshape(-1, 1))
                    labels[labels == 0] = -1
                elif up_method == 'op':
                    clustering = OPTICS(max_eps=eps, min_samples=min_samples)
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
            except:
                return 'N'
                
            threshold = kwargs.get('cluster_size_threshold', 0.2)
            one_cluster_rule = kwargs.get('one_cluster_rule', False)
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            unique_labels = zip(unique_labels, unique_counts)
            unique_labels = list(filter(lambda x: x[0] != -1, unique_labels))
            if len(unique_labels) == 0:
                return 'N'
            if one_cluster_rule and len(unique_labels) > 1:
                return 'N'
            x, y = map(list, zip(*unique_labels))
            max_count = max(y)
            if max_count / len(possible_predictions) < threshold:
                return 'N'

            max_cluster = list(filter(lambda x: x[1] == max_count, unique_labels))[0]
            
            avg = np.mean(np.array(possible_predictions)[labels == max_cluster[0]])
            if random_perturbation:
                avg += np.random.normal(0, 0.01)
            return avg
        
    def predict(self, preceding_array, h, non_pred_model, up_method = 'db', **kwargs):
        
        n = len(preceding_array)
        k = self.max_pattern_len - n

        start_i = 1
        if h > self.max_pattern_len - 2:
            raise ValueError('Horizon > max possible len of prediction')


        predictions = []
        possible_predictions = [[] for i in range(k + n - start_i - 1)]

        weight = kwargs.get('weight', 5)
        repeat = kwargs.get('repeat', 1)

        for i in range(start_i, self.max_pattern_len - max(k, 0)):

            for _ in range((i // weight + 1) * repeat):
            
                 pred_array = self.continue_trajectory(preceding_array[-i:])
            
                 for j in range(len(pred_array[(start_i + i):])):

                     possible_predictions[j].append(pred_array[(start_i + i):][j])
                
        for i in range(len(possible_predictions)):
            if non_pred_model.is_predictable(possible_predictions[i]):
                predictions.append(self.unified_prediction(possible_predictions[i], up_method, random_perturbation=False, **kwargs))
            else:
                predictions.append('N')

        return predictions[:h], possible_predictions[:h]



class predictor_markov_chain_gradient():
    def __init__(self):
        self.graph = None
        self.train_array = None
        self.max_pattern_len = None

    def fit_train_array(self, train_array):
        self.train_array = train_array

    def produce_graph(self, max_pattern_len = 100, round_ = 1, mode = 'ind'):
        
        if self.train_array.all() == None:
            raise ValueError('Train array is empty')

        if len(self.train_array) < max_pattern_len:
            raise ValueError('Max pattern length is greater than train array length')

        G = nx.DiGraph()
        n = len(self.train_array)
        G.add_node(0, grad = None, level = None)

        for i in range(n - max_pattern_len):
            current_node = 0

            for j in range(max_pattern_len - 2):
                grad = round(self.train_array[i + j] - 2 * self.train_array[i + j + 1] + self.train_array[i + j + 2], round_)

                if mode == 'uni':
                    current_level = [i for i in G.nodes if G.nodes[i]['level'] == j + 1]
                elif mode == 'ind':
                    current_level = [i for i in G.neighbors(current_node)]

                if grad not in [G.nodes[i]['grad'] for i in current_level]:
                    new_node = len(G)
                    G.add_node(new_node, grad = grad, level = j + 1)
                    G.add_edge(current_node, new_node, weight = 1)
                    current_node = new_node
            
                else:
                    correct_node = [i for i in current_level if G.nodes[i]['grad'] == grad][0]

                    if G.has_edge(current_node, correct_node):
                        G[current_node][correct_node]['weight'] += 1
                
                    else:
                        G.add_edge(current_node, correct_node, weight = 1)
                
                    current_node = correct_node
        
        self.graph = G
        self.max_pattern_len = max_pattern_len
        pass

    def get_graph(self):
        return self.graph
    
    def get_max_pattern_len(self):
        return self.max_pattern_len

    def get_trajectory(self, start_couple, trajectory_len, power = 1):

        if self.graph == None:
            raise ValueError('Graph not founded')

        if trajectory_len > self.max_pattern_len:
            raise ValueError('Trajectory length > Max possible trajectory lenght')

        trajectory = [start_couple[0], start_couple[1]]

        velocity = trajectory[0] - trajectory[1]
        current_y = trajectory[1]
        current_node = 0

        for i in range(trajectory_len - 3):
            
            weights = np.array([self.graph[current_node][i]['weight'] for i in self.graph.neighbors(current_node)]) ** power
            nodes = [i for i in self.graph[current_node]]
            weights = weights / weights.sum()

            next_node = np.random.choice(nodes, p = weights)
            velocity = velocity - self.graph.nodes[next_node]['grad']
            current_y = current_y - velocity
            trajectory.append(current_y)
            current_node = next_node

        return trajectory
    
    def continue_trajectory(self, trace, power = 1):
        trajectory = [trace[0], trace[1]]
        velocity = trace[0] - trace[1]
        current_node = 0

        for i in range(len(trace) - 2):
            grad = trace[i] - 2 * trace[i + 1] + trace[i + 2]
            nodes = [(i, self.graph.nodes[i]['grad']) for i in self.graph.neighbors(current_node)]
            error = [(i, np.sqrt(j - grad)) for i, j in nodes]
            next_node = min(error, key = lambda x: x[1])[0]

            velocity = velocity - self.graph.nodes[next_node]['grad']
            trajectory.append(trajectory[-1] - velocity)
            current_node = next_node
        
        for i in range(self.max_pattern_len - len(trace)):
            weights = np.array([self.graph[current_node][i]['weight'] for i in self.graph.neighbors(current_node)]) ** power
            nodes = [i for i in self.graph[current_node]]
            weights = weights / weights.sum()

            next_node = np.random.choice(nodes, p = weights)
            velocity = velocity - self.graph.nodes[next_node]['grad']
            trajectory.append(trajectory[-1] - velocity)
            current_node = next_node
        
        return trajectory

    def unified_prediction(self, possible_predictions, up_method, \
        random_perturbation=False, **kwargs):
        '''Calculates unified prediciton from set of possible predicted values

        Parameters
        ----------
        possible_predictions: list or 1D np.array
            List of possible predictions
        up_method: str from {'a', 'wi', 'db', 'op'}
            Method of estimating unified prediciton
            'a'  - average
            'wi' - clustering with Wishart, get largest cluster mean
            'db' - clustering with DBSCAN, get largest cluster mean
            'op' - clustering with OPTICS, get largest cluster mean
        random_perturbation: boolean
            Add noise to unified prediction

        **kwargs
        -- for Wishart, DBSCAN and OPTICS clustering --
        min_samples: int > 1 or float between 0 and 1, default=5
            Minimal number of samples in cluster
        eps: float from 0 to 1, default=0.01
            Max distance within one cluster
        cluster_size_threshold: float from 0 to 1, default=0.2
            Minimal percentage of points in largest cluster to call point predictable
        one_cluster_rule: boolean, defalut=False
            Point is predictable only is there is one cluster (not including noise)
        '''

        if len(possible_predictions) == 0:
            return 'N'
        min_samples = kwargs.get('min_samples', 5)
        eps = kwargs.get('eps', 0.01)

        if up_method == 'a':
            avg = np.mean(possible_predictions)
            if random_perturbation:
                avg += np.random.normal(0, 0.01)
            return avg

        if up_method == 'wi' or up_method == 'db' or up_method=='op':  
            try: 
                if up_method == 'db': 
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
                    
                elif up_method == 'wi':
                    clustering = Wishart(min_samples, eps)
                    labels = clustering.fit(np.array(possible_predictions).reshape(-1, 1))
                    labels[labels == 0] = -1
                elif up_method == 'op':
                    clustering = OPTICS(max_eps=eps, min_samples=min_samples)
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
            except:
                return 'N'
                
            threshold = kwargs.get('cluster_size_threshold', 0.2)
            one_cluster_rule = kwargs.get('one_cluster_rule', False)
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            unique_labels = zip(unique_labels, unique_counts)
            unique_labels = list(filter(lambda x: x[0] != -1, unique_labels))
            if len(unique_labels) == 0:
                return 'N'
            if one_cluster_rule and len(unique_labels) > 1:
                return 'N'
            x, y = map(list, zip(*unique_labels))
            max_count = max(y)
            if max_count / len(possible_predictions) < threshold:
                return 'N'

            max_cluster = list(filter(lambda x: x[1] == max_count, unique_labels))[0]
            
            avg = np.mean(possible_predictions[labels == max_cluster[0]])
            if random_perturbation:
                avg += np.random.normal(0, 0.01)
            return avg
        
    def predict(self, preceding_array, h, non_pred_model, up_method = 'a'):
        
        n = len(preceding_array)
        k = self.max_pattern_len - n

        start_i = 2
        if h > self.max_pattern_len - 3:
            raise ValueError('Horizon > max possible len of prediction')

        predictions = []
        possible_predictions = [[] for i in range(k + n - start_i - 1)]

        for i in range(start_i, self.max_pattern_len - max(k, 0)):
            
            pred_array = self.continue_trajectory(preceding_array[-i:])
            
            for j in range(len(pred_array[(start_i + i):])):

                possible_predictions[j].append(pred_array[(start_i + i):][j])
                
        for i in range(len(possible_predictions)):
            if non_pred_model.is_predictable(possible_predictions[i]):
                predictions.append(self.unified_prediction(possible_predictions[i], up_method))
            else:
                predictions.append('N')

        return predictions[:h], possible_predictions[:h]





class NonPredModel():
    def is_predictable(self, possible_predictions, **kwargs):
        pass
    
    def reset(self):
        pass

    def __str__(self):
        return self.__class__.__name__


class ForcedPredictionNPM(NonPredModel):
    def __init__(self):
        pass

    def is_predictable(self, possible_predictions):
        return True

    def __str__(self):
        return self.__class__.__name__

class RapidGrowthNPM(NonPredModel):
    def __init__(self):
        self.min_max_spreads = []

    def is_predictable(self, possible_predictions):
        if len(possible_predictions) == 0:
            return False
        current_spread = max(possible_predictions) - min(possible_predictions)
        self.min_max_spreads.append(current_spread)
        self.min_max_spreads = self.min_max_spreads[-4:]
        if len(self.min_max_spreads) < 4:
            return True
        if self.min_max_spreads[0] < self.min_max_spreads[1] \
            and self.min_max_spreads[1] < self.min_max_spreads[2] \
                and self.min_max_spreads[2] < self.min_max_spreads[3]:
            return False
        else:
            return True
    
    def reset(self):
        self.min_max_spreads = []

    def __str__(self):
        return self.__class__.__name__


class RapidGrowthDBSCANNPM(NonPredModel):
    def __init__(self, min_samples=5, eps=0.01):
        self.dbscan_spreads = []
        self.min_samples = min_samples
        self.eps = eps

    def is_predictable(self, possible_predictions):
        if len(possible_predictions) == 0:
            return False
        dbscan = DBSCAN(min_samples=self.min_samples, eps=self.eps)
        try:
            labels = dbscan.fit_predict(np.array(possible_predictions).reshape(-1, 1))
        except:
            labels = [-1]
        unique_clusters = np.unique(labels)
        self.dbscan_spreads.append(len(unique_clusters))
        self.dbscan_spreads = self.dbscan_spreads[-4:]
        if len(unique_clusters) == 1 and -1 in unique_clusters:
            return False
        if len(self.dbscan_spreads) < 4:
            return True
        if self.dbscan_spreads[0] < self.dbscan_spreads[1] \
            and self.dbscan_spreads[1] < self.dbscan_spreads[2] \
                and self.dbscan_spreads[2] < self.dbscan_spreads[3]\
                    and self.dbscan_spreads[3] > 2:
            return False
        else:
            return True

    def reset(self):
        self.dbscan_spreads = []

    def __str__(self):
        return self.__class__.__name__


class RapidGrowthWishartNPM(NonPredModel):
    def __init__(self, min_samples=5, eps=0.01):
        self.wishart_spreads = []
        self.min_samples = min_samples
        self.eps = eps

    def is_predictable(self, possible_predictions):
        if len(possible_predictions) == 0:
            return False
        wishart = Wishart(self.min_samples, self.eps)
        try:
            labels = wishart.fit(np.array(possible_predictions).reshape(-1, 1))
        except:
            labels = [0] 
        unique_clusters = np.unique(labels)
        self.wishart_spreads.append(len(unique_clusters))
        self.wishart_spreads = self.wishart_spreads[-4:]
        if len(unique_clusters) == 1 and 0 in unique_clusters:
            return False
        if len(self.wishart_spreads) < 4:
            return True
        if self.wishart_spreads[0] < self.wishart_spreads[1] \
            and self.wishart_spreads[1] < self.wishart_spreads[2] \
                and self.wishart_spreads[2] < self.wishart_spreads[3]\
                    and self.wishart_spreads[3] > 2:
            return False
        else:
            return True

    def reset(self):
        self.wishart_spreads = []

    def __str__(self):
        return self.__class__.__name__

class LimitClusterSizeNPM(NonPredModel):
    def __init__(self, min_cluster_size, max_n_clusters):
        self.min_cluster_size = min_cluster_size
        self.max_n_clusters = max_n_clusters


    def is_predictable(self, possible_predictions):
        dbscan = DBSCAN(eps=0.01, min_samples=5)
        try:
            labels = dbscan.fit_predict(np.array(possible_predictions).reshape(-1, 1))
        except:
            return False
        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        unique_labels = zip(unique_labels, unique_counts)
        unique_labels = list(filter(lambda x: x[0] != -1, unique_labels))
        if len(unique_labels) == 0:
            return False
        if len(unique_labels) > self.max_n_clusters:
            return False
        x, y = map(list, zip(*unique_labels))
        max_count = max(y)
        if max_count / len(possible_predictions) < self.min_cluster_size:
            return False
        return True

    def __str__(self):
        return self.__class__.__name__ + '(min_cluster_size=' + \
            str(self.min_cluster_size) + ', max_n_clusters=' + \
            str(self.max_n_clusters) + ')'


class BigLeapBtwIterationsNPM(NonPredModel):
    def __init__(self, base_non_pred_model : NonPredModel = None):
        self.max_leap = 0.2
        self.base_non_pred_model = base_non_pred_model


    def is_predictable(self, possible_predictions):
        if self.base_non_pred_model is None:
            return True
        return self.base_non_pred_model.is_predictable(possible_predictions)


    def is_predictable_by_up_log(self, up_log):
        if len(up_log) < 2:
            return True
        current_up = up_log[-1]
        if current_up == 'N':
            return False
        last_known_up = up_log[-2]
        j = 2
        while j <= min(len(up_log) - 1, 4) and last_known_up == 'N':
            j += 1
            last_known_up = up_log[-j]
        if last_known_up == 'N':
            return True
        if abs(current_up - last_known_up) > self.max_leap:
            return False
        return True

    
    def reset(self):
        if self.base_non_pred_model is not None:
            self.base_non_pred_model.reset()

    def __str__(self):
        return self.__class__.__name__ + '(base_non_pred_model=' \
            + str(self.base_non_pred_model) + ')'


class WeirdPatternsNPM(NonPredModel):
    def __init__(self, Y1, eps0=0.1, base_non_pred_model : NonPredModel = None):
        self.base_non_pred_model = base_non_pred_model
        self.patterns = [[1, 1, 1], [1, 2, 1]]
        self.clustered_motifs = []
        self.eps0 = eps0
        for pattern in self.patterns:
            motifs = []

            X_cl_idx = len(Y1) - 1 - np.cumsum(pattern[::-1])
            X_cl_idx = X_cl_idx[::-1]
            X_cl_idx = np.append(X_cl_idx, len(Y1) - 1)
            x = [X_cl_idx - i for i in range(len(Y1) - np.sum(pattern))]
            
            X_cl = np.array([np.take(Y1, X_cl_idx - i) for i in range(len(Y1) - np.sum(pattern))])
            X_cl = X_cl.astype(float)

            dbscan = DBSCAN(eps=0.01, min_samples=5, metric='euclidean')
            cl_labels = dbscan.fit_predict(X_cl)
            
            n_clusters = len(np.unique(cl_labels))
            if np.isin(cl_labels, -1).any():
                n_clusters -= 1
            if n_clusters == 0:
                motifs = []
            else:
                motifs = np.array([np.mean(X_cl[cl_labels  == i], axis=0) for i in range(n_clusters)])
            self.clustered_motifs.append(motifs)


    def is_predictable(self, possible_predictions):
        if self.base_non_pred_model is None:
            return True
        return self.base_non_pred_model.is_predictable(possible_predictions)

    def is_predictable_by_up(self, unified_predictions):
        h = len(unified_predictions)
        is_pred = [True] * h
        for j in range(len(self.patterns)):
            pattern = self.patterns[j]
            for i in range(h - sum(pattern)):
                idx = [i, i + pattern[0], i + pattern[0] + pattern[1],\
                    i + pattern[0] + pattern[1] + pattern[2]]
                to_check = np.take(unified_predictions, idx)
                if 'N' in to_check:
                    continue
                to_check = to_check.astype(float)
                match_found = False
                for motif in self.clustered_motifs[j]:
                    if np.linalg.norm(motif - to_check) < self.eps0:
                        match_found = True
                        break
                if not match_found:
                    for k in [i + pattern[0], i + pattern[0] + pattern[1],\
                        i + pattern[0] + pattern[1] + pattern[2]]:
                        is_pred[k] = False

        return is_pred

    def __str__(self):
        return self.__class__.__name__ + '(base_non_pred_model=' \
            + str(self.base_non_pred_model) + ', eps0=' + str(self.eps0) + ')'