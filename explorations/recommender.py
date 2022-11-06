from abc import abstractmethod
from sklearn.neighbors import NearestNeighbors

class BaseRecommender():
    @abstractmethod
    def get_top_recommendations(k): 
        pass 

class knn(BaseRecommender):
    def __init__(self): 
        self.distances = None
        self.indices = None

    def fit(self, X, **kwargs): 
        nbrs = NearestNeighbors(**kwargs).fit(X)
        self.distances, self.indices = nbrs.kneighbors(X)

    def get_top_recommendations(self, df, row_idx): 
        return df.filter(self.indices[row_idx], axis=0)