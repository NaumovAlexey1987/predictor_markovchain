import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from itertools import product, combinations
import pickle
from scipy.stats import moment, kurtosis, entropy
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from scipy.special import gamma

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


class NonPredModel():
    def is_predictable(self, possible_predictions, **kwargs):
        pass
    
    def reset(self):
        pass


class ForcedPredictionNPM(NonPredModel):
    def __init__(self):
        pass

    def is_predictable(self, possible_predictions):
        return True


class LargeSpreadNPM(NonPredModel):
    def __init__(self, pred_model, kappa=1., \
        k_max=10, pattern_len=3, up_method='a', Y3=[]):
        '''
        Parameters
        ----------
        pred_model: TimeSeriesPredictor
            Prediction model
        kappa: float, default=1.0
            Coefficient before average spread value when evaluating 
            whether point is predictable
        k_max: int
            Max distance between observations in motif
        pattern_length: int
            Length of pattern
        up_method: str from {'a', 'db', 'wi', 'op'}, default='a'
            Method of calculating unified prediction
        Y3: list or 1D np.ndarray, default=[]
            Validation set
        '''
        avg = .0
        Y_pred = []
        not_empty = 0
        for i in range(k_max*pattern_len + 1, len(Y3)):
            possible_preds_val, _ = pred_model.predict_one_step(Y3[:i], \
                Y_pred, match_thershold=0.01)
            if len(possible_preds_val) > 0:
                avg += np.max(possible_preds_val) - np.min(possible_preds_val)
                not_empty += 1
            Y_pred.append(pred_model.unified_prediction(possible_preds_val, up_method))
        self.ls_average = avg / not_empty
        self.kappa = kappa


    def __init__(self, ls_average, kappa):
        self.ls_average = ls_average
        self.kappa = kappa


    def is_predictable(self, possible_predictions):
        ''' Evaluates whether point is predictable

        Parameters
        ----------
        possible_predictions: list or 1D np.array
            List of possible predictions
        '''
        if len(possible_predictions) == 0:
            return False
        return max(possible_predictions) - min(possible_predictions) < self.ls_average * self.kappa
        

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


class LogRegNPM(NonPredModel):
    def __init__(self, Y3, predictor0, mc_method='db', beta=0.2):
        Y_preceding = Y3[:31]
        eps_close = 0.05
        features = []
        labels = []
        for i in range(32, len(Y3) - 1):
            Y_preceding = Y3[:i - 1]
            possible_predictions, _ = predictor0.predict_one_step(Y_preceding, [])
            unified_pred = predictor0.unified_prediction(possible_predictions, 'db')

            stats = self.get_stats(possible_predictions)
            if stats is not None:
                if unified_pred != 'N' and abs(unified_pred - Y3[i]) < eps_close:
                    labels.append(1)
                else:
                    labels.append(0)
                features.append(stats)
        features = np.array(features)

        self.logreg = LogisticRegression()
        self.logreg.fit(features, labels)


    def is_predictable(self, possible_predictions):
        stats = np.array(self.get_stats(possible_predictions)).reshape(1, -1)
        if stats is None:
            return False
        res = self.logreg.predict(stats)
        if res == 0:
            return False
        return True

    def get_stats(self, possible_predictions):
        if len(possible_predictions) > 0:
            clustering_labels = DBSCAN(eps=0.01, min_samples=5).fit_predict(np.array(possible_predictions).reshape(-1, 1))
            unique, counts = np.unique(clustering_labels, return_counts=True)
            if unique[0] == -1:
                unique = unique[1:]
                counts = counts[1:]
            n_clusters = len(unique)

            if n_clusters == 0:
                largest_cluster_rel_size = 0
                diff = 0
            else:
                largest_cluster_rel_size = max(counts) / len(possible_predictions)
                if len(unique) > 1:
                    diff = max(counts) / len(possible_predictions) - min(counts) / \
                        len(possible_predictions)
                else:
                    diff = 1
            

            return [moment(possible_predictions, 2), moment(possible_predictions, 3), \
                    moment(possible_predictions, 4), entropy(possible_predictions), \
                    kurtosis(possible_predictions), n_clusters, largest_cluster_rel_size, diff]
        else:
            return None 


class BigLeapNPM(NonPredModel):
    def __init__(self, Y3, base_non_pred_model : NonPredModel = None):
        diffs = np.ediff1d(Y3)
        self.min_leap = min(diffs) * 0.95
        self.max_leap = max(diffs) * 1.05
        self.base_non_pred_model = base_non_pred_model


    def is_predictable(self, possible_predictions):
        if self.base_non_pred_model is None:
            return True
        return self.base_non_pred_model.is_predictable(possible_predictions)


    def is_predictable_by_up(self, unified_predictions):
        diffs = []
        for i in range(len(unified_predictions) - 1):
            try:
                diffs.append(abs(float(unified_predictions[i]) \
                    - float(unified_predictions[i + 1])))
            except ValueError:
                diffs.append(-1) 
        is_pred = [True \
            if diffs[i] == -1 or (diffs[i] < self.max_leap and diffs[i] > self.min_leap) \
            else False \
            for i in range(len(diffs))]
        is_pred.insert(0, True)
        return is_pred


    def reset(self):
        if self.base_non_pred_model is not None:
            self.base_non_pred_model.reset()


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


class BigLeapBtwIterationsNPM(NonPredModel):
    def __init__(self, base_non_pred_model : NonPredModel = None):
        self.max_leap = 0.2
        self.base_non_pred_model = base_non_pred_model


    def is_predictable(self, possible_predictions):
        if self.base_non_pred_model is None:
            return True
        return self.base_non_pred_model.is_predictable(possible_predictions)


    def is_predictable_by_up_log(self, up_log):
        # up_log for one point
        # if a jump btw last unified pred and current > max_leap --> non_pred
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


class WeirdPatternsNPM(NonPredModel):
    def __init__(self, Y1, eps0=0.1, base_non_pred_model : NonPredModel = None):
        self.base_non_pred_model = base_non_pred_model
        self.patterns = [[1, 1, 1], [1, 2, 1]]
        self.clustered_motifs = []
        self.eps0 = eps0
        for pattern in self.patterns:
            motifs = []

            # make dataset for classification
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


class TimeSeriesPredictor:
    '''Class for time series prediction 
    Usage : 
        -> Create `NonPredModel` instance
        -> Create `TimeSeriesPredictor` instance with `clustered_motifs` from file and `NonPredModel`
        -> If `clustered_motifs` do not exist, run `cluster_motifs` method
        -> Run `predict` method to predict time series with base algorithm
        -> Run `self_healing` method
    Attributes
    ----------
    clustered_motifs : list of size (n_patterns)
        List of tuples of clustered motifs : [(pattern, clusters)]
        clusters : list of length (n_patterns) of np.array of shape 
            (n_clusters, pattern_length + 1)
    non_pred_model : NonPredModel
        Model for identifying non-predictable points, see `non_pred_model` module
    k_max : int, default=10
        Max distance between points in pattern
    pattern_length : int, default=3
        Length of patterns
    '''


    def __init__(self, clustered_motifs, non_pred_model, k_max=10, pattern_length=3):
        '''Constructor
        If provided `clustered_motifs` is None, run `cluster_motifs` method

        Parameters
        ----------
        clustered_motifs : list of size (n_patterns) or None
            List of tuples of clustered motifs : [(pattern, clusters)]
            clusters : list of length (n_patterns) of np.array of shape 
                (n_clusters, pattern_length + 1)
        non_pred_model : NonPredModel
            Model for identifying non-predictable points
        k_max : int, default=10
            Max distance between points in pattern
        pattern_length : int, default=3
            Length of patterns
        '''
        self.clustered_motifs = clustered_motifs     
        self.non_pred_model = non_pred_model
        self.k_max = k_max
        self.pattern_length = pattern_length


    def cluster_motifs(self, Y1, beta=0.1, mc_method='db', \
        k_max=10, pattern_length=3, **kwargs):
        '''Cluster motifs by patterns, result saved in class attributes and returned from function
        n_patterns = beta * k_max^(pattern_length+1)

        Parameters
        ----------
        Y1 : list or 1d np.ndarray
            Training time series data
        beta : float from 0 to 1, default=0.1
            Percentage of used patterns
        mc_method : {'wi', 'db'}, default='db'
            Clustering method for motifs; 'wi' - Wishart, 'db' - DBSCAN
        k_max : int, default=10
            Max distance between points in pattern
        pattern_length : int, default=3
            Length of patterns
        
        **kwargs : dict
        eps : float from 0 to 1, default=0.01
            Max distance within one cluster for DBSCAN and Wishart clustering 
        min_samples : int > 1 or float from 0 to 1, default=5
            Min number of samples in cluster for DBSCAN and Wishart clustering 

        Returns
        -------
        clustered_motifs : list of size (n_patterns)
            List of tuples of clustered motifs : [(pattern, clusters)]
            clusters : list of length (n_patterns) of np.array of shape 
                (n_clusters, pattern_length + 1)
        '''
        clustered_motifs = []

        self.k_max = k_max
        self.pattern_length = pattern_length
        patterns = list(product(range(1, k_max + 1), repeat=pattern_length))
        patterns = np.array([list(p) for p in patterns])
        
        used_patterns = patterns[np.random.choice(patterns.shape[0], \
            size=int(beta * patterns.shape[0]))]

        eps = kwargs.get('eps', 0.01)
        min_samples = kwargs.get('min_samples', 5)

        for pattern in used_patterns:
            motifs = []

            # make dataset for classification
            X_cl_idx = len(Y1) - 1 - np.cumsum(pattern[::-1])
            X_cl_idx = X_cl_idx[::-1]
            X_cl_idx = np.append(X_cl_idx, len(Y1) - 1)
            x = [X_cl_idx - i for i in range(len(Y1) - np.sum(pattern))]
            
            X_cl = np.array([np.take(Y1, X_cl_idx - i) for i in range(len(Y1) - np.sum(pattern))])
            X_cl = X_cl.astype(float)

            if mc_method == 'db':
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                cl_labels = dbscan.fit_predict(X_cl)
            elif mc_method == 'wi':
                wishart = Wishart(min_samples, eps)
                cl_labels = wishart.fit(X_cl)
                cl_labels = cl_labels - 1
            
            n_clusters = len(np.unique(cl_labels))
            if np.isin(cl_labels, -1).any():
                n_clusters -= 1
            if n_clusters == 0:
                motifs = []
            else:
                motifs = np.array([np.mean(X_cl[cl_labels  == i], axis=0) for i in range(n_clusters)])
            clustered_motifs.append(motifs)
        
        self.clustered_motifs = list(zip(used_patterns, clustered_motifs))

        return self.clustered_motifs
              
                
    def set_motifs(self, clustered_motifs, k_max=10, pattern_length=3):
        '''Set used patterns and motifs, usually loaded from file

        Parameters
        ----------
        clustered_motifs : list of size (n_patterns)
            List of tuples of clustered motifs : [(pattern, clusters)]
            clusters : list of length (n_patterns) of np.array of shape 
                (n_clusters, pattern_length + 1)
        k_max : int, default=10
            Max distance between points in pattern
        pattern_length : int, default=3
            Length of patterns
        '''
        self.clustered_motifs = clustered_motifs
        self.k_max = k_max
        self.pattern_length = pattern_length
    

    def set_non_pred_model(self, non_pred_model : NonPredModel):
        '''Sets NonPredModel to class attribute

        Parameters
        ----------
        non_pred_model : NonPredModel
            Model for identifying non-predictable points
        '''
        self.non_pred_model = non_pred_model


    def predict(self, Y_preceding, h, up_method, \
        alg_type, match_thershold=0.01, **kwargs):
        '''Predict h next values of time series

        Parameters
        ----------
        Y_preceding : list or 1D np.array
            Time series segment preceding the segment to predict. 
            Size has to be at least k_max*pattern_length (30 for default parameters)
        h : int
            Prediciton horizon
        up_method : str from {'a', 'wi', 'db', 'op'}
            Method of estimating unified prediction
        alg_type : str from {'s', 'tp'}
            's' - one-step predicition, no trajectories, p = 1
            'tp' - trajectories with random perturbation, p > 1
        match_thershold : float, default=0.01
            Threshold for motif to match

        **kwargs
        n_trajectories : int, default=20
            Number of trajectories in case `alg_type`='tp'

        Returns
        -------
            Y_pred : np.ndarray
                Predicted values, 'N' for non-predictable point
            possible_predictions_list : list
                List of possible predicted values for each point to be predicted
            trajectories : list
                Predicted trajectories in case `alg_type`='tp'
        '''
        
        if alg_type == 's':
            Y_pred = np.array([], dtype=object)
            possible_predictions_list = []
            for i in range(h):
                possible_predictions, distances = self.predict_one_step(Y_preceding, \
                    Y_pred, match_thershold)

                is_predictable = self.non_pred_model.is_predictable(possible_predictions)
                if is_predictable:
                    kwargs['distances'] = distances
                    unified_prediction = self.unified_prediction(possible_predictions, \
                        up_method, **kwargs)
                else:
                    unified_prediction = 'N'
                
                Y_pred = np.append(Y_pred, unified_prediction)
                possible_predictions_list.append(possible_predictions)
            
            if hasattr(self.non_pred_model, 'is_predictable_by_up'):
                is_pred = self.non_pred_model.is_predictable_by_up(Y_pred)
                for i in range(h):
                    if not is_pred[i]:
                        Y_pred[i] = 'N'

            self.non_pred_model.reset()
            return Y_pred, possible_predictions_list

        if alg_type == 'tp':
            n_trajectories = kwargs.get('n_trajectories', 20)
            trajectories = []
            for i in range(n_trajectories):
                Y_pred = np.array([])
                for i in range(h):
                    possible_predictions, distances = self.predict_one_step(Y_preceding, \
                        Y_pred, match_thershold)

                    is_predictable = self.non_pred_model.is_predictable(possible_predictions)
                    if is_predictable:
                        kwargs['distances'] = distances
                        kwargs['random_perturbation'] = True
                        unified_prediction = self.unified_prediction(possible_predictions, \
                            up_method, **kwargs)
                    else:
                        unified_prediction = 'N'
                    
                    Y_pred = np.append(Y_pred, unified_prediction)
                trajectories.append(Y_pred)
            
            trajectories = np.array(trajectories)
            Y_pred = np.array([])
            possible_predictions_list = []
            for i in range(h):
                possible_predictions = trajectories[:, i]
                possible_predictions = possible_predictions[possible_predictions != 'N']
                possible_predictions = possible_predictions.astype(float)
                possible_predictions_list.append(possible_predictions)
                # unified_prediction = self.unified_prediction(possible_predictions, up_method, **kwargs)
                unified_prediction = np.mean(possible_predictions)
                Y_pred = np.append(Y_pred, unified_prediction)
            return Y_pred, possible_predictions_list, trajectories

    
    def predict_one_step(self, Y_preceding, Y_pred, match_thershold=0.01):
        '''One step prediction of time series

        Parameters
        ----------
        Y_preceding : list or 1D np.array
            Time series segment preceding the segment to predict. 
            Size has to be at least k_max*pattern_length (30 for default parameters)
        Y_pred : list or 1D np.ndarray
            Already predicted values from previous steps of algorithm
        match_thershold : float, default=0.01
            Threshold for motif to match

        Returns
        -------
        possible_predictions : np.ndarray
            List of possible predictions for the point
        distances : np.ndarray
            List of distances of possible prediction to cluster center, 
            used for some algorithms of calculating unified prediction
        '''
        Y_all = np.append(Y_preceding, Y_pred)
        possible_predictions = np.array([])
        distances = np.array([])
        
        for i in range(len(self.clustered_motifs)):
            pattern = self.clustered_motifs[i][0]
            motifs = self.clustered_motifs[i][1]

            c_idx = np.full(self.pattern_length, len(Y_all))
            for j in range(self.pattern_length):
                c_idx[j] -= np.sum(pattern[j:])
            
            c = np.take(Y_all, c_idx, axis=0)
            if np.isin(c, 'N').any():
                continue
            c = c.astype(float)
            
            for motif in motifs:
                if np.linalg.norm(c - motif[:-1]) < match_thershold:
                    possible_predictions = np.append(possible_predictions, motif[-1])
                    distances = np.append(distances, np.linalg.norm(c - motif[:-1]))
        return possible_predictions, distances
   

    def unified_prediction(self, possible_predictions, up_method, \
        random_perturbation=False, **kwargs):
        '''Calculates unified prediciton from set of possible predicted values

        Parameters
        ----------
        possible_predictions : list or 1D np.array
            List of possible predictions
        up_method : str from {'a', 'wi', 'db', 'op'}
            Method of estimating unified prediciton
            'a'  - average
            'wi' - clustering with Wishart, get largest cluster mean
            'db' - clustering with DBSCAN, get largest cluster mean
            'op' - clustering with OPTICS, get largest cluster mean
        random_perturbation : boolean
            Add noise to unified prediction

        **kwargs
        -- for Wishart, DBSCAN and OPTICS clustering --
        min_samples : int > 1 or float between 0 and 1, default=5
            Minimal number of samples in cluster
        eps : float from 0 to 1, default=0.01
            Max distance within one cluster

        Returns
        -------
        avg : float or 'N'
            Unified predicted value, 'N' for non-predictable points
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

        if up_method == 'wi' or up_method == 'db' or up_method=='op' :  
            try : 
                if up_method == 'db' : 
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
                
            unique_labels, unique_counts = np.unique(labels, return_counts=True)
            unique_labels = zip(unique_labels, unique_counts)
            unique_labels = list(filter(lambda x : x[0] != -1, unique_labels))
            if len(unique_labels) == 0:
                return 'N'
            x, y = map(list, zip(*unique_labels))
            max_count = max(y)

            max_cluster = list(filter(lambda x : x[1] == max_count, unique_labels))[0]
            
            avg = np.mean(possible_predictions[labels == max_cluster[0]])
            if random_perturbation:
                avg += np.random.normal(0, 0.01)
            return avg


    def unified_prediction_weighted(self, possible_predictions, sep_indices, \
        up_method, weight_method='double_clustering', return_pp_without_noise=False, \
        **kwargs):
        '''Calculates weighted unified prediciton from set of possible predicted values

        Parameters
        ----------
        possible_predictions : list or 1D np.array
            List of possible predictions
        sep_indices : list
            List of inidices which separate iterations in `possible_predictions`
        up_method : str from {'wi', 'db', 'op'}
            Method of estimating unified prediciton
            'wi' - clustering with Wishart, get largest cluster mean
            'db' - clustering with DBSCAN, get largest cluster mean
            'op' - clustering with OPTICS, get largest cluster mean
        weight_method : str from {'double_clustering', 'weighred_average', 'factor', 
            'pattern_length', 'pattern_length_dist', 'dist', 'dist_factor'}
            Method of calculating weights of points
        return_pp_without_noise : boolean
            If True returns `possible_predictions` without noise cluster

        **kwargs
        -- for Wishart, DBSCAN and OPTICS clustering --
        min_samples : int > 1 or float between 0 and 1, default=5
            Minimal number of samples in cluster
        eps : float from 0 to 1, default=0.01
            Max distance within one cluster

        Returns
        -------
        avg : float or 'N'
            Unified predicted value, 'N' for non-predictable points
        '''
        
        if weight_method == 'double_clustering' or weight_method == 'weighted_average':
            ups = []
            for i in range(len(sep_indices) - 1):
                pp_iteration = possible_predictions[sep_indices[i]:sep_indices[i+1]]
                ups.append(self.unified_prediction(pp_iteration, up_method=up_method, \
                    random_perturbation=False, **kwargs))
            ups = np.array(ups, dtype=object)
            
            predictable = np.argwhere(ups != 'N').reshape(1, -1)[0]
            predictable_ups = np.take(ups, predictable).astype(float)
            eps_dist_clusters = 0.02
            if weight_method == 'double_clustering':
                if len(predictable_ups) == 0:
                    return 'N'
                if len(predictable_ups) <= 2:
                    return predictable_ups[0]
            
                if len(predictable_ups) > 2:
                    dbscan2 = DBSCAN(min_samples=2, eps=0.1)
                    labels2 = dbscan2.fit_predict(predictable_ups.reshape(-1, 1))
                    unique_labels, unique_counts = np.unique(labels2, return_counts=True)
                    unique_labels = zip(unique_labels, unique_counts)
                    unique_labels = list(filter(lambda x : x[0] != -1, unique_labels))
                    if len(unique_labels) == 0:
                        return predictable_ups[0]
                    x, y = map(list, zip(*unique_labels))
                    max_count = max(y)

                    max_cluster = list(filter(lambda x : x[1] == max_count, unique_labels))[0]
                    
                    avg = np.mean(predictable_ups[labels2 == max_cluster[0]])
                    return avg
            if weight_method == 'weighted_average':
                factor = 0.9
                weights = [factor ** x for x in range(len(sep_indices) - 1)]
                return np.average(predictable_ups, weights[:len(predictable_ups)])

        if weight_method in ['pattern_length_dist', 'factor', \
            'pattern_length', 'dist', 'dist_factor']:
            eps = kwargs.get('eps', 0.01)
            min_samples= kwargs.get('min_samples', 5)
            try : 
                if up_method == 'db' : 
                    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
                elif up_method == 'wi':
                    clustering = Wishart(min_samples, eps)
                    labels = clustering.fit(np.array(possible_predictions).reshape(-1, 1))
                    labels = labels - 1
                elif up_method == 'op':
                    clustering = OPTICS(max_eps=eps, min_samples=min_samples)
                    labels = clustering.fit_predict(np.array(possible_predictions).reshape(-1, 1))
            except:
                return 'N'
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            factor = kwargs.get('factor', 0.9)
            if weight_method == 'factor':
                point_weights = []
                for i in range(len(sep_indices) - 1):
                    point_weights.extend([factor ** i] * (sep_indices[i + 1] - sep_indices[i]))
            elif weight_method == 'dist_factor':
                point_weights = kwargs.get('point_weights')
                for i in range(len(sep_indices) - 1):
                    for k in range(sep_indices[i], sep_indices[i + 1]):
                        point_weights[k] *= factor ** i
            else:
                point_weights = kwargs.get('point_weights')
                
            unique_labels = unique_labels if -1 not in unique_labels else unique_labels[1:]
            if len(unique_labels) == 0:
                return 'N'
            cluster_size = [0] * len(unique_labels)
            # print(point_weights)
            # print(sep_indices)
            for i in range(len(labels)):
                # print(len(cluster_size), labels[i])
                cluster_size[labels[i]] += point_weights[i]
            max_cluster = np.argmax(cluster_size)

            avg = np.mean(possible_predictions[labels == unique_labels[max_cluster]])

            if return_pp_without_noise:
                return avg, possible_predictions[labels != -1]
            return avg

        
    def self_healing_one_iteration(self, Y_preceding, unified_predictions, \
        possible_predictions, h, up_method='db', weighted=False, sep_indices=[], \
        fixed_points_idx=[], **kwargs):
        '''One iteration of self-healing 

        Parameters
        ----------
        Y_preceding : list or 1D np.array
            Time series segment preceding the segment to predict. 
            Size has to be at least k_max*pattern_length (30 for default parameters)
        unified_predictions : list or 1D np.ndarray of length h
            List of unified predictions from previous iteration or from base algorithm
        possible_predictions : list of h np.ndarrays
            List of lists of possible predictions for each point from previous uteration 
            or from base algorithm
        h : int
            Prediction horison
        up_method : str from {'a', 'wi', 'db', 'op'}
            See method unified_prediction
        weighted : boolean, deafult=False
            Calculate weighted unified prediction or not
        sep_indices : list
            List of inidices which separate iterations in `possible_predictions`
        fixed_points_idx : list
            List of points which do not change their status

        **kwargs
        weight_method : str from {'double_clustering', 'weighted_average', 'factor', 
        'pattern_length', 'pattern_length_dist', 'dist', 'dist_factor'}
            Method of calculating weights of points


        Returns
        ----------
        new_up : np.ndarray
            List of new unified predictions
        possible_predictions : list of np.ndarrays
            List of old and new possible predictions
        sep_indices : list
            List of inidices which separate iterations in `possible_predictions`, 
            if unified prediction is calculated with weights
        '''

        def take_ordered(indices, n_to_take):
            if n_to_take == 0:
                return [[]]
            return [list(c) for c in list(combinations(indices, n_to_take))]
            

        def get_all_matching_patterns(sliced):
            '''Get all patterns with not-N unified predictions

            Parameters
            ----------
            sliced : list or 1D np.ndarray
                List of unified predictions 30 before and 30 after (or to prediction horison) 
                predicted point

            Returns
            ----------
            matched_patterns : list
                List of tuples (j, index), where j is place of predicted point in pattern, 
                index is index of pattern in self.used_patterns
            '''
            used_patterns = [x[0] for x in self.clustered_motifs]
            matched_patterns = []
            current_point_index = self.k_max * self.pattern_length
            pred_left = np.argwhere(sliced[:current_point_index] != 'N').reshape(1, -1)[0]
            pred_right = np.argwhere(sliced[current_point_index + 1:] != 'N').reshape(1, -1)[0] + current_point_index + 1
            
            for left_n_of_points in range(self.pattern_length + 1):
                right_n_of_points = self.pattern_length - left_n_of_points
                take_left = take_ordered(pred_left.tolist(), left_n_of_points)
                
                for i in range(len(take_left)):
                    take_left[i].append(current_point_index)
                take_right = take_ordered(pred_right.tolist(), right_n_of_points)
                combos = [x[0] + x[1] for x in list(product(take_left, take_right))]
                dists = np.diff(combos)
                for dist in dists:
                    if np.all((dist > 0) & (dist <= self.k_max)):
                        index = -1
                        for i in range(len(used_patterns)):
                            if np.array_equal(used_patterns[i].tolist(), dist):
                                index = i
                                break
                        if index >= 0:
                            matched_patterns.append(tuple([left_n_of_points, index]))
            
            return matched_patterns


        Y_prec_with_up = np.append(Y_preceding, unified_predictions)
        healing_motif_match_eps = kwargs.get('healing_motif_match_eps', 0.01)
        pp_weights = kwargs.get('pp_weights')
        weight_method = kwargs.get('weight_method', None)
        for i in range(h):
            left = len(Y_preceding) + i - self.k_max * self.pattern_length
            right = len(Y_preceding) + i + self.k_max * self.pattern_length
            if right >= len(Y_prec_with_up):
                right = len(Y_prec_with_up) - 1
            curr_point_idx = self.k_max * self.pattern_length
            sliced = np.array(Y_prec_with_up[left:right + 1])

            indices_patterns_to_check = get_all_matching_patterns(sliced)
            for (j, index) in indices_patterns_to_check:
                pattern = self.clustered_motifs[index][0]
                motifs = self.clustered_motifs[index][1]

                c_idx = np.full(self.pattern_length + 1, curr_point_idx)
                c_idx[j] = curr_point_idx
                for k in range(j):
                    c_idx[k] -= np.sum(pattern[k:j])
                for k in range(j+1,self.pattern_length+1):
                    c_idx[k] += np.sum(pattern[j:k])
                
                try:
                    c = np.take(sliced, c_idx, axis=0)
                except:
                    continue
                c = np.delete(c, j, axis=0)
                if 'N' in c:
                    continue
                c = c.astype(float)
            
                for motif in motifs:
                    cut_motif = np.delete(motif, j)
                    if np.linalg.norm(c - cut_motif) < healing_motif_match_eps:
                        possible_predictions[i] = np.append(possible_predictions[i], \
                            motif[j])
                        if weight_method == 'pattern_length':
                            pp_weights[i].append(1 / sum(pattern))
                        if weight_method == 'pattern_length_dist':
                            pp_weights[i].append((1 / sum(pattern)) * (1 / np.linalg.norm(c - cut_motif)))
                        if weight_method == 'dist' or weight_method == 'dist_factor':
                            pp_weights[i].append(1 / np.linalg.norm(c - cut_motif))
                        
            if weighted:
                sep_indices[i].append(len(possible_predictions[i]))
        new_up = []
        for i in range(len(possible_predictions)):
            if self.non_pred_model.is_predictable(possible_predictions[i]) or \
                i in fixed_points_idx:

                if not weighted:
                    new_up.append(self.unified_prediction(possible_predictions[i], \
                        up_method=up_method, **kwargs))
                else:
                    if kwargs.get('clear_noise', False):
                        u, possible_predictions[i] = self.unified_prediction_weighted(\
                            possible_predictions[i], \
                            up_method=up_method, \
                            sep_indices=sep_indices[i], \
                            return_pp_without_noise=True, \
                            point_weights=pp_weights[i], \
                            **kwargs)
                        new_up.append(u)
                    else:
                        new_up.append(self.unified_prediction_weighted(\
                            possible_predictions[i], \
                            up_method=up_method, \
                            sep_indices=sep_indices[i], point_weights=pp_weights[i], \
                            **kwargs))
            else:
                new_up.append('N')
        
        if hasattr(self.non_pred_model, 'is_predictable_by_up'):
            is_pred = self.non_pred_model.is_predictable_by_up(new_up)
            for i in range(h):
                if not is_pred[i]:
                    new_up[i] = 'N'

        self.non_pred_model.reset()
        if not weighted:
            return np.array(new_up, dtype=object), possible_predictions
        else:
            return np.array(new_up, dtype=object), possible_predictions, sep_indices


    def self_healing(self, Y_preceding, h, return_n_iterations=False, \
        eps_stop=0.01, unified_predictions=None, \
        possible_predictions=None, healing_up_method='db', weighted_up=False,
        fixed_points_idx=[], healing_logs_filepath=None, \
        **kwargs):
        '''Iterative algorithm of self-healing
        Criteria of stopping : 3 iterations no new unified predictions found AND 
            euclidian distance between two last iterations is less than eps_stop

        Parameters
        ----------
        Y_preceding : list or 1D np.array
            Time series segment preceding the segment to predict. 
            Size has to be at least k_max*pattern_length (30 for default parameters)
        h : int
            Prediction horison
        unified_predictions : list or 1D np.ndarray of length h, default=None
            List of unified predictions from base algorithm
            Can be None, then first iteration will be the same as base algorithm
        possible_predictions : list of h np.ndarrays
            List of lists of possible predictions for each point from base algorithm
            Can be None, then first iteration will be the same as base algorithm
        healing_up_method : str from {'a', 'wi', 'db', 'op'}
            See method unified_prediction
        weighted : boolean, deafult=False
            Calculate weighted unified prediction or not
        weight_method : str from {'double_clustering', 'weighted_average', 'factor', 
        'pattern_length', 'pattern_length_dist', 'dist', 'dist_factor'}
            Method of calculating weights of points
        fixed_points_idx : list
            List of points which do not change their status

        **kwargs
        See method unified_prediction

        Returns
        ----------
        unified_predictions : np.ndarray
            List of unified predictions after self-healing algorithm
        '''
        if unified_predictions is None:
            unified_predictions = np.full(h, 'N', dtype=object)
        if possible_predictions is None:
            possible_predictions = [[] for _ in range(h)]

        up_logs = []
        pp_logs = []
        non_pred_logs = []
        iteration = 0
        stop_criteria = False
        up_logs.append(unified_predictions.copy())
        pp_logs.append(possible_predictions.copy())
        non_pred_logs.append(np.count_nonzero(unified_predictions == 'N'))
        sep_indices = [[0] for _ in range(h)]
        used_patterns = [x[0] for x in self.clustered_motifs]
        pl = [sum(pattern) for pattern in used_patterns]
        avg_pattern_length = sum(pl) / len(pl)
        pp_weights = [[1 / avg_pattern_length] * len(possible_predictions[i]) for i in range(h)]
        kwargs['pp_weights'] = pp_weights

        while not stop_criteria:
            iteration += 1
            if not weighted_up:
                unified_predictions, possible_predictions = \
                    self.self_healing_one_iteration(Y_preceding, \
                    unified_predictions, possible_predictions, h, \
                    up_method=healing_up_method, fixed_points_idx=fixed_points_idx, \
                    **kwargs)
            else:
                unified_predictions, possible_predictions, sep_indices = \
                    self.self_healing_one_iteration(Y_preceding, \
                    unified_predictions, possible_predictions, h, up_method=healing_up_method, \
                        weighted=True, sep_indices=sep_indices, \
                        fixed_points_idx=fixed_points_idx, **kwargs)
            
            
            if hasattr(self.non_pred_model, 'is_predictable_by_up_log'):
                for i in range(h):
                    up_log = [l[i] for l in up_logs]
                    up_log.append(unified_predictions[i])
                    if not self.non_pred_model.is_predictable_by_up_log(up_log):
                        unified_predictions[i] = 'N'
                        

            up_logs.append(unified_predictions.copy())
            pp_logs.append(possible_predictions.copy())
            non_pred_logs.append(np.count_nonzero(unified_predictions == 'N'))

            
            if len(up_logs) > 2:
                np_1 = np.argwhere(up_logs[-1] != 'N').reshape(1, -1)[0]
                np_2 = np.argwhere(up_logs[-2] != 'N').reshape(1, -1)[0]
                np_3 = np.argwhere(up_logs[-3] != 'N').reshape(1, -1)[0]

                if np.array_equal(np_1, np_2) and np.array_equal(np_2, np_3):
                    pred_indices = np.argwhere(up_logs[-1] != 'N')
                    predicted_1 = np.take(up_logs[-1], pred_indices)
                    predicted_2 = np.take(up_logs[-2], pred_indices)
                    if 'N' not in predicted_2:
                        if np.linalg.norm(predicted_1.astype(float) - \
                            predicted_2.astype(float)) < eps_stop:
                            stop_criteria = True
        
        if healing_logs_filepath is not None:
            with open(healing_logs_filepath, 'wb') as f:
                pass
            with open(healing_logs_filepath, 'ab') as f:
                pickle.dump(up_logs, f)
                pickle.dump(pp_logs, f)
                pickle.dump(non_pred_logs, f)

        if return_n_iterations:
            return unified_predictions, possible_predictions, iteration
        else:
            return unified_predictions, possible_predictions

   