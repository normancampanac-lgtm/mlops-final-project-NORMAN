# setup.py (versión mínima recomendada)
from setuptools import setup

setup(
    name="mlops_housing",
    version="1.0.0",
    packages=["src"],
    install_requires=[
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "fastapi",
        "mlflow",
    ],
)