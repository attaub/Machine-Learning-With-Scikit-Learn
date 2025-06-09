import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv("./datasets/housing/housing.csv")
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50, figsize=(20, 15))
plt.show()
