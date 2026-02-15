import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hummingbird_kanish.dataloader import DataLoader
from hummingbird_kanish.optimizer import Optimizer

def generate_mock_movie_data():
    """Generates a dummy dataset for movie preferences."""
    np.random.seed(42)
    data = {
        'genre_action': np.random.randint(0, 2, 100),
        'genre_comedy': np.random.randint(0, 2, 100),
        'genre_drama': np.random.randint(0, 2, 100),
        'release_year_2020plus': np.random.randint(0, 2, 100),
        'is_blockbuster': np.random.randint(0, 2, 100),
        'avg_runtime_over_2h': np.random.randint(0, 2, 100),
        'user_liked': np.random.randint(0, 2, 100) # Target variable
    }
    df = pd.DataFrame(data)
    df.to_csv("movie_data.csv", index=False)

if __name__ == "__main__":
    #Preparing data
    generate_mock_movie_data()
    
    #initialize library 
    print("starting AHO-based feature selection")
    data_loader = DataLoader("./movie_data.csv")
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    optimizer = Optimizer(rf_classifier, dataset=data_loader)

    #run optimization
    results = optimizer.fit()

    #display results
    print("\n--- Optimization Results ---")
    print("Most important features for recommendation:")
    sorted_features = sorted(results["columns"], key=lambda x: x["score"], reverse=True)
    
    for feat in sorted_features:
        status = "SELECTED" if feat['score'] > 0 else "DROPPED"
        print(f"{status} | Feature: {feat['name']} (Score: {feat['score']})")