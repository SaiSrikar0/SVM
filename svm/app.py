import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#creating a synthetic dataset
from sklearn.datasets import make_classification

s, y = make_classification(n_samples = 1000, n_features = 2, n_classes = 2, n_clusters_per_class = 2, n_redundant = 0)
