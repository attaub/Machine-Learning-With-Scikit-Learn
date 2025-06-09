###########################################
from data_utils import save_fig, fetch_housing_data, load_housing_data
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# model_selection
# base.BaseEstimator and base.TransformerMixin
# impute.SimpleImputer
# preprocessing.OrdinalEncoder, OneHotEncoder
# pipeline
# compose.ColumnTransformer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('axes', labelsize=24)
mpl.rc('xtick', labelsize=22)
mpl.rc('ytick', labelsize=22)

fetch_housing_data()
housing = load_housing_data()
housing.head()  # 10 attributes
housing.info()  # 20,640 instances, 20,433 non null values in total_betrooms
# ocean_proximity is non numerical

housing["ocean_proximity"].value_counts()
# INLAND:9136, NEAR OCEAN:6551, NEAR BAY:2290, ISLAND:5

housing.describe()
# count, mean, min, max, std, 25% 50% 75%

housing.hist(bins=50, figsize=(20, 15))
save_fig("attribute_histogram_plots")
plt.show()

# medina_income: capped, -> 30,000: not a problem, but you should know it
# media_age median_value: capped, check with clinet team if this is a problem. may get the proper values for districts with capped values or remove them from the training set and test set

# different scale of fearuters
# tail heavy histograms: transform to bell shapped curve

# Create test set:  Avoid data snooping, 80 to 20 ratio

# if randomly created, but next time it will pick a different subset and eventually will see the entire the dataset

# Sol: save test set on the first run or use random seed
# will break next time if dataset is updated

# solution: use instance identifiers, compute hash of each instance's idientifier and put the instance into training set if the hash <= 20% of the max hash value.

# bad news: no identifier column in the dataset
# simplest solution: use row index as the ID, but make sure the new data gets appended in the end of the dataest

# You can also use most stable features to build unique identifers

#  Geolocation coordinates are unique


##################################################################
##################################################################
# train_test split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# this is purely random sampling. Fine for large enough datasets
# otherwise introduces significan sampling bias
# tain set must be representative of the entire population
# In our case, if the client asks to make sure test set is representative of the various income categories.

housing["median_income"].hist()
plt.show()

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

housing["income_cat"].value_counts()
housing["income_cat"].hist()
plt.show()


ss_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in ss_split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


#############################################
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)


#############################################
# measure the income category proportions in the full dataset.
# test set with stratified sampling has income category proportions almost identical to those in the full dataset
# test set with random sampling is skewed.


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#############################################
# check if test set representative of dataset  

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()

compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)

compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

compare_props


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

##################################################################
##################################################################
##################################################################
# Discover and Visualize the data to gain insights


##################################################################
# no validation in this case
housing = strat_train_set.copy()

# Visualize Geograhpical data
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
plt.show()


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
plt.show()

#
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False,
)
plt.legend()
save_fig("housing_prices_scatterplot")
plt.show()


##
# housing price are correlated to location and population density
# clustering algorithm to add new features
# ocean proximity is useful


############################################
# Looking for correlations
#
######################
# Select only numeric columns
housing_numerical = housing.select_dtypes(include=['number'])
corr_matrix = housing_numerical.corr()
# corr_matrix.hist()
# plt.show()

corr_matrix["median_house_value"].sort_values(ascending=False)


#
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
# from pandas.plotting import scatter_matrix

attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
plt.show()


housing.plot(
    kind="scatter", x="median_income", y="median_house_value", alpha=0.1
)

plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")
plt.show()


# experimenting with attribute aombinations

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]

housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)

housing["population_per_household"] = (
    housing["population"] / housing["households"]
)

housing_numerical = housing.select_dtypes(include=['number'])
corr_matrix = housing_numerical.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# from pandas.tools.plotting import (scatter_matrix,)  # For older versions of Pandas
# from pandas.plotting import scatter_matrix

attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]

pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")


housing.plot(
    kind="scatter", x="median_income", y="median_house_value", alpha=0.1
)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")


# Experimenting with Attribute Combinations
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)
housing["population_per_household"] = (
    housing["population"] / housing["households"]
)


housing_numerical = housing.select_dtypes(include=['number'])
# corr_matrix["median_house_value"].sort_values(ascending=False)
corr_matrix = housing_numerical.corr()


housing_numerical.plot(
    kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2
)
plt.axis([0, 5, 0, 520000])
plt.show()

housing.describe()


#######################################################################
#######################################################################
#######################################################################
# Prepare the Data for Machine Learning Algorithms

# drop labels for training set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


########################################################################
# Data Cleaning


sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

sample_incomplete_rows.dropna(subset=["total_bedrooms"])  # option 1
sample_incomplete_rows.drop("total_bedrooms", axis=1)  # option 2

median = housing["total_bedrooms"].median()

sample_incomplete_rows["total_bedrooms"].fillna(
    median, inplace=True
)  # option 3
sample_incomplete_rows

# do it the better way
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")


housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

housing_tr.loc[sample_incomplete_rows.index.values]
imputer.strategy

housing_tr = pd.DataFrame(
    X, columns=housing_num.columns, index=housing_num.index
)
housing_tr.head()


########################################################################
# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Ordinalencoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_

# Onehotencoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()

# cat_encoder = OneHotEncoder(sparse=False)
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
cat_encoder.categories_

########################################################################
# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args, **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do # required for sklearn pipelines

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room,
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


col_names = "total_rooms", "total_bedrooms", "population", "households"

#  get the column indices
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names
]

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)
    + ["rooms_per_household", "population_per_household"],
    index=housing.index,
)

housing_extra_attribs.head()


######################################################################
# Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ]
)

# housing_num_tr = num_pipeline.fit_transform(housing_num)
# housing_num_tr


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)

# housing_num = housing.drop("ocean_proximity", axis=1)
# num_attribs = list(housing.drop("ocean_proximity", axis=1))
# cat_attribs = ["ocean_proximity"]

full_pipeline_2 = ColumnTransformer(
    [
        ("num", num_pipeline, list(housing.drop("ocean_proximity", axis=1))),
        ("cat", OneHotEncoder(), ["ocean_proximity"]),
    ]
)
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared_2 = full_pipeline_2.fit_transform(housing)

housing_prepared.shape


# For reference, here is the old solution based on a DataFrameSelector transformer (to just select a subset of the Pandas DataFrame columns), and a FeatureUnion:

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# old_num_pipeline = Pipeline([
#         ('selector', OldDataFrameSelector(num_attribs)),
#         ('imputer', SimpleImputer(strategy="median")),
#         ('attribs_adder', CombinedAttributesAdder()),
#         ('std_scaler', StandardScaler()),
#     ])

# old_cat_pipeline = Pipeline([
#         ('selector', OldDataFrameSelector(cat_attribs)),
#         ('cat_encoder', OneHotEncoder(sparse=False)),
#     ])
# from sklearn.pipeline import FeatureUnion

# old_full_pipeline = FeatureUnion(transformer_list=[
#         ("num_pipeline", old_num_pipeline),
#         ("cat_pipeline", old_cat_pipeline),
#     ])
# old_housing_prepared = old_full_pipeline.fit_transform(housing)
# old_housing_prepared


# np.allclose(housing_prepared, old_housing_prepared)

############################################
############################################
############################################
# Select and Train a Model
## Training and Evaluating on the Training Set

######################
# starting with linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

## try full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
some_labels = housing_labels.iloc[:5]
lin_reg.predict(some_data_prepared)

list(some_labels)
some_data_prepared



housing_predictions = lin_reg.predict(housing_prepared)

# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
lin_rmse

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae


######################
######################
# Decision Trees
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Better Evaluation Using Cross-Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)

tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)


lin_scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)

######################
# Random Forests
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


forest_scores = cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)
pd.Series(np.sqrt(-scores)).describe()


############################################
# Support Vector Machines
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

##########################################################################
##########################################################################
##########################################################################
## Fine-Tune Your Model

# Grid Search
from sklearn.model_selection import GridSearchCV

# Parameters Grid
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# Random Forset: Parameter Tuning
forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

grid_search.best_estimator_

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)

############################################
############################################
# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
)

rnd_search.fit(housing_prepared, housing_labels)


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.round(np.sqrt(-mean_score),5), params)

# Analyze the Best Models and Their Errors
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# Evaluate Your System on the Test Set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

# We can compute a 95% confidence interval for the test RMSE:

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(
    stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
)


m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)


# Extra material
# A full pipeline with both preparation and prediction
full_pipeline_with_predictor = Pipeline(
    [("preparation", full_pipeline), ("linear", LinearRegression())]
)

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)
array(
    [
        210644.60459286,
        317768.80697211,
        210956.43331178,
        59218.98886849,
        189747.55849879,
    ]
)
# Model persistence using joblib
my_model = full_pipeline_with_predictor
import joblib

joblib.dump(my_model, "my_model.pkl")  # DIFF
# ...

my_model_loaded = joblib.load("my_model.pkl")  # DIFF

# Example SciPy distributions for RandomizedSearchCV

from scipy.stats import geom, expon

geom_distrib = geom(0.5).rvs(10000, random_state=42)
expon_distrib = expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()
