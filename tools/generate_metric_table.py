import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('test_metrics.csv')

for k, v in df.items():
    data = v.to_numpy()
    print("{}: {:.3f}±{:.3f}".format(k, data.mean() * 100, data.std() * 100))

# TODO: Complete this
