import numpy as np
from sklearn.model_selection import cross_val_score

class AHOSelector:
    def __init__(self, model, n_birds=10, max_iter=15):
        self.model = model
        self.n_birds = n_birds
        self.max_iter = max_iter

    def fitness(self, solution, X, y):
        selected_indices = [i for i, bit in enumerate(solution) if bit == 1]
        if len(selected_indices) == 0: return 0
        
        X_subset = X.iloc[:, selected_indices]
        # Cross-validation for stability
        scores = cross_val_score(self.model, X_subset, y, cv=3)
        return scores.mean()

    def run(self, X, y):
        n_features = X.shape[1]
        # Initialization
        population = [np.random.randint(0, 2, n_features) for _ in range(self.n_birds)]
        best_sol = population[0]
        best_fit = -1

        for _ in range(self.max_iter):
            for i in range(self.n_birds):
                fit = self.fitness(population[i], X, y)
                if fit > best_fit:
                    best_fit = fit
                    best_sol = population[i].copy()
        
        # Final scores for all features
        feature_scores = np.zeros(n_features)
        for i, val in enumerate(best_sol):
            if val == 1: feature_scores[i] = best_fit
            
        return best_sol, feature_scores