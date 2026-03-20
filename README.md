# Hummingbird-AHO: Feature Selection Library 🐦



A specialized Python library implementing **Artificial Hummingbird Optimization (AHO)** for high-dimensional feature selection. This project operationalizes the research presented at the **ANTIC 2025 Conference (Springer CCIS)**.

## 🚀 Overview
Hummingbird-AHO provides a production-ready interface to significantly reduce feature dimensionality while maximizing model performance. It abstracts complex metaheuristic optimization into a simple, high-level API designed for rapid deployment in machine learning pipelines.



## 📦 Installation
Clone the repository and install it as a local package:
```bash
git clone https://github.com/KanishDhiman02/Hummingbird_AHO_Optimizer_Engine.git
cd Hummingbird_AHO_Optimizer_Engine
pip install .
```
## 🛠️ Quick Start
The library is designed to be plug-and-play with scikit-learn estimators.
```bash
from optimizer_engine import DataLoader, Optimizer
from sklearn.ensemble import RandomForestClassifier

# 1. Load your dataset
data = DataLoader("./your_dataset.csv")

# 2. Initialize the optimizer (Works with any sklearn classifier)
rf_classifier = RandomForestClassifier(random_state=42)
optimizer = Optimizer(rf_classifier, dataset=data)

# 3. Run the AHO Feature Selection
results = optimizer.fit()

# 4. Results analysis
print("Optimized Feature Subset:")
for feat in results["columns"]:
    if feat["score"] > 0:
        print(f"✅ {feat['name']}")
```
## 🎬 Practical Demo (Movie Recommendation)
The repository includes a practical use-case demonstrating how AHO can optimize a recommendation engine by filtering non-predictive metadata.

To run the demo:
```bash
python examples/movie_recommendation_demo.py
```
## 🧬 Key Engineering Features
Modular Architecture: Clean separation of data handling, optimization core, and evaluation logic.

High Performance: Optimized for speed and low memory footprint in high-dimensional spaces.

SDE-Ready: Full support for virtual environments and standard Python packaging (setup.py).

Research-Backed: Engineered to operationalize advanced feature-reduction strategies.
