from .aho_logic import AHOFeatureSelector

class Optimizer:
    def __init__(self, estimator, dataset, n_hummingbirds=20, max_iter=50):
        self.dataset = dataset
        self.selector = AHOFeatureSelector(
            estimator=estimator, 
            n_hummingbirds=n_hummingbirds, 
            max_iter=max_iter
        )

    def fit(self):

        self.selector.fit(self.dataset.X, self.dataset.y)
        
        results = {"columns": []}

        for i, name in enumerate(self.dataset.feature_names):
            results["columns"].append({
                "name": name,
                "score": 1.0 if self.selector.support_[i] else 0.0
            })
        return results