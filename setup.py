from setuptools import setup, find_packages

setup(
    name="hummingbird-kanish",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn"
    ],
    author="Kanish Dhiman",
    description="AHO algorithm for feature selection in machine learning pipelines.",
    url="https://github.com/KanishDhiman02/hummingbird-kanish",
)
