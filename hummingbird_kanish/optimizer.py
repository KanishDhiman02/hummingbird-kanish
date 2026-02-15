from .aho_logic import AHOSelector

class Optimizer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.selector = AHOSelector(model)

    def fit(self):
        best_bits, scores = self.selector.run(self.dataset.X, self.dataset.y)
        
        results = {"columns": []}
        for i, name in enumerate(self.dataset.feature_names):
            results["columns"].append({
                "name": name,
                "score": round(float(scores[i]), 4)
            })
        return results