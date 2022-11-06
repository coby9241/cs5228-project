import numpy as np

from abc import abstractmethod
from sklearn.neighbors import NearestNeighbors

class BaseRecommender():
    @abstractmethod
    def get_top_recommendations(k): 
        pass 

class knn(BaseRecommender):
    def __init__(self, df, X_transformed): 
        self.df = df
        self.X_transformed = X_transformed
        self.model = None

    def get_top_recommendations(self, row, k=5, feature_idx='default', return_different_property=False, 
                                max_k=None, degree_of_randomisation=None, refit_model=False, **kwargs):
        
        # All features are used to fit model if users do not specify preferred features
        if feature_idx == 'default':
            feature_idx = range(0, self.X_transformed.shape[1])
        
        # Decide n_neighbours based on k
        n_neighbors = k + 1 if max_k is None else max_k + 1

        # Fit model if model does not exist or user wants to refit
        if self.model is None or refit_model:
            if degree_of_randomisation:
                if max_k < k:
                    raise ValueError('max_k must be larger than k. Please specify it correctly.')
                if degree_of_randomisation <= 0 or degree_of_randomisation > 1:
                    raise ValueError('degree_of_randomisation must be between 0 and 1. Please specify it correctly.')

            # Fit model
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, **kwargs).fit(self.X_transformed[:, feature_idx])
            self.model = nbrs

        distances, indices = self.model.kneighbors(self.X_transformed[row.index[0], feature_idx].reshape(1, -1), n_neighbors=self.df.shape[0])
        distances, indices = distances[0], indices[0]
        indices = np.delete(indices, np.argwhere(indices == row.index[0]))

        if return_different_property:
            property_name = row['property_name'].iloc[0]
            exclude_index = self.df[self.df['property_name']==property_name].index
            indices = indices[~np.isin(indices, exclude_index) ]

        if degree_of_randomisation:
            indices = indices[:n_neighbors]
            random_idx = np.random.choice(indices[k + 1:], int(np.ceil(k * degree_of_randomisation)), replace=False)
            top_idx = indices[:(k + 1) - len(random_idx)]
            concat_idx = np.concatenate((top_idx, random_idx), axis=None)
            output_idx = np.delete(concat_idx, np.argwhere(concat_idx == row.index[0]))
            return self.df.filter(output_idx[:k], axis=0)

        return self.df.filter(indices[:k], axis=0)