# Hummingbird-AHO: Feature Selection Library 🐦

A specialized Python library implementing **Artificial Hummingbird Optimization (AHO)** for feature selection in machine learning pipelines. This project operationalizes research originally presented at the **ANTIC 2025 Conference (Springer CCIS)**.

##  Overview
The library provides an automated way to reduce feature dimensionality by simulating the foraging and territorial behaviors of hummingbirds. It utilizes a weighted fitness function to balance classification accuracy and feature reduction, ensuring optimal performance for complex datasets.

##  Installation
Clone the repository and install it as a local package:
```bash
git clone [https://github.com/KanishDhiman02/hummingbird-kanish.git](https://github.com/KanishDhiman02/hummingbird-kanish.git)
cd hummingbird-kanish
pip install .

Quick Start 
The library is designed to be plug-and-play with scikit-learn estimators.
from hummingbird_kanish.dataloader import DataLoader
from hummingbird_kanish.optimizer import Optimizer
from sklearn.ensemble import RandomForestClassifier

# 1. Load your dataset
data = DataLoader("./your_dataset.csv")

# 2. Initialize the optimizer (works with any sklearn-compatible classifier)
rf_classifier = RandomForestClassifier(random_state=42)
optimizer = Optimizer(rf_classifier, dataset=data)

# 3. Run the AHO Feature Selection
results = optimizer.fit()

# 4. Results analysis
print("Optimized Feature Subset:")
for feat in results["columns"]:
    if feat["score"] > 0:
        print(f" {feat['name']}")

Practical Demo (Movie Recommendation)
To demonstrate the "operationalization" of this research for real-world scenarios, I have included a movie recommendation use-case. This script generates synthetic user-preference data and uses AHO to identify the most predictive features (genres, ratings, etc.).

To run the demo: python examples movie_recommendation_demo.py
