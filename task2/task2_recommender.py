import numpy as np

from abc import abstractmethod
from sklearn.neighbors import NearestNeighbors

class BaseRecommender():
    @abstractmethod
    def get_top_recommendations(k): 
        pass 

class knn(BaseRecommender):
    def __init__(self, df, X_transformed, pipe): 
        self.df = df
        self.X_transformed = X_transformed
        self.scaler = pipe
        self.model = None

    def get_top_recommendations(self, row, k=5, feature_idx='default', max_k=None,
                                degree_of_randomisation=None, refit_model=False, **kwargs):
        
        # All features are used to fit model if users do not specify preferred features
        if feature_idx == 'default':
            feature_idx = range(0, self.X_transformed.shape[1])

        if self.model is None or refit_model:
            # Decide n_neighbours based on k
            n_neighbors = k + 1 if max_k is None else max_k + 1
            if degree_of_randomisation is not None:
                if max_k < k:
                    raise ValueError('max_k must be larger than k. Please specify it correctly.')
                if degree_of_randomisation <= 0 or degree_of_randomisation > 1:
                    raise ValueError('degree_of_randomisation must be between 0 and 1. Please specify it correctly.')

            # Fit model
            input_X = self.X_transformed[:, feature_idx]
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, **kwargs).fit(input_X)

        row_transformed = self.scaler.transform(row)
        distances, indices = nbrs.kneighbors(row_transformed[:, feature_idx], n_neighbors=n_neighbors)

        if degree_of_randomisation:
            random_idx = np.random.choice(indices[0][k + 1:], int(np.ceil(k * degree_of_randomisation)), replace=False)
            top_idx = indices[0][:(k + 1) - len(random_idx)]
            concat_idx = np.concatenate((top_idx, random_idx), axis=None)
            output_idx = np.delete(concat_idx, np.argwhere(concat_idx == row.index[0]))
            return self.df.filter(output_idx, axis=0)
        
        output_idx = np.delete(indices[0], np.argwhere(indices[0] == row.index[0]))
        return self.df.filter(output_idx, axis=0)
