import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

class AHOFeatureSelector:
    def __init__(self, estimator, n_hummingbirds=20, max_iter=50, alpha=0.99, verbose=True, random_state=42):
        self.estimator = estimator
        self.n_hummingbirds = n_hummingbirds
        self.max_iter = max_iter
        self.alpha = alpha
        self.verbose = verbose
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _fitness(self, solution, X, y):
        selected_features = np.where(solution == 1)[0]
        if len(selected_features) == 0:
            return 1.0

        X_subset = X[:, selected_features] if isinstance(X, np.ndarray) else X.iloc[:, selected_features]
        
        scores = cross_val_score(self.estimator, X_subset, y, cv=3, scoring='accuracy')
        accuracy = np.mean(scores)
        error_rate = 1 - accuracy
        
        feature_reduction_rate = len(selected_features) / X.shape[1]

        return self.alpha * error_rate + (1 - self.alpha) * feature_reduction_rate

    def fit(self, X, y):

        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if isinstance(y, pd.Series) else y
        n_features = X_vals.shape[1]
        
        positions = np.random.rand(self.n_hummingbirds, n_features)
        fitness = np.full(self.n_hummingbirds, np.inf)
        visit_table = np.zeros((self.n_hummingbirds, self.n_hummingbirds))
        
        for i in range(self.n_hummingbirds):
            binary_sol = (self._sigmoid(positions[i, :]) > 0.5).astype(int)
            fitness[i] = self._fitness(binary_sol, X_vals, y_vals)
        
        best_fitness = np.min(fitness)
        best_idx = np.argmin(fitness)
        self.best_solution_ = (self._sigmoid(positions[best_idx, :]) > 0.5).astype(int)

        for t in range(1, self.max_iter + 1):
            for i in range(self.n_hummingbirds):
                target_idx = np.random.randint(0, self.n_hummingbirds)
                while target_idx == i: target_idx = np.random.randint(0, self.n_hummingbirds)
                
                visit_table[i, target_idx] += 1
                
                D = np.random.choice([-1, 1]) * np.random.rand()
                new_pos = positions[target_idx, :] + D * (positions[i, :] - positions[target_idx, :])
                
                if np.random.rand() < 0.5:
                    positions[i, :] = new_pos
                else:
                    positions[i, :] = positions[i, :] + np.random.rand() * positions[i, :]
                
                positions[i, :] = np.clip(positions[i, :], -10, 10)
                binary_sol = (self._sigmoid(positions[i, :]) > 0.5).astype(int)
                curr_fit = self._fitness(binary_sol, X_vals, y_vals)
                
                if curr_fit < fitness[i]:
                    fitness[i] = curr_fit
                    if curr_fit < best_fitness:
                        best_fitness = curr_fit
                        self.best_solution_ = binary_sol

            migration_idx = np.argmax(np.sum(visit_table, axis=1))
            positions[migration_idx, :] = np.random.rand(n_features)
            visit_table[migration_idx, :] = 0 
            
            if self.verbose and t % 10 == 0:
                print(f"Iteration {t}: Best Fitness = {best_fitness:.4f}")
        
        self.support_ = self.best_solution_.astype(bool)
        return self

    def transform(self, X):
        return X.iloc[:, self.support_] if isinstance(X, pd.DataFrame) else X[:, self.support_]