import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


housing = pd.read_csv("./datasets/housing/housing.csv")

train_set, test_set = train_test_split(housing, test_size=0.2)

housing = train_set.drop("median_house_value", axis=1)
housing_num = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ]
)

# cat_pipeline = Pipeline([('onehotencoder', OneHotEncoder())])

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)

housing_prepared = full_pipeline.fit_transform(housing)

#################################################################
lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

some_labels = housing_labels.iloc[:5]

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_predictions, housing_labels)
lin_mae = mean_absolute_error(housing_predictions, housing_labels)

print(np.sqrt(lin_mse), lin_mae)


#################################################################
#################################################################
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

print("#" * 50)
print(tree_rmse)
print("#" * 50)


#################################################################
#################################################################
## Better Evaluation Using Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


print("#" * 50)


def display_scores(scores):
    print()
    print("=" * 78)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("=" * 78)
    print()


print("#" * 50)

scores = cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)

tree_rmse_scores = np.sqrt(-scores)

lin_scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10,
)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(tree_rmse_scores)
display_scores(lin_rmse_scores)

#################################################################
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

"""
def cross_val_scor(model, X,y,scrring="mse",cv=10):
    return mean(mse)
"""
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

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)

print("=" * 50)
print(svm_rmse)
print("=" * 50)

#################################################################
###   Fine-Tune Your Model
# Grid Search

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid = [
    {
        'n_estimators': [3, 10, 30],
        'max_features': [2, 4, 6, 8],
    },
    {
        'bootstrap': [False],
        'n_estimators': [3, 10],
        'max_features': [2, 3, 4],
    },
]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
)


"""
def my_grid_search(my_model, list_of_dict_of_prms, cv, scroring):
    pass
"""

grid_search.fit(housing_prepared, housing_labels)

#################################################################
# The best hyperparameter combination found:

grid_search.best_params_
grid_search.best_estimator_

print("#" * 40)
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Estimator:", grid_search.best_estimator_)
print("#" * 40)

# Let's look at the score of each hyperparameter combination tested during the grid search:

cvres = grid_search.cv_results_
print()
print("=" * 50)
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print("=" * 50)
print()

pd.DataFrame(grid_search.cv_results_)
print("=" * 50)
print(pd.DataFrame(grid_search.cv_results_))
print("=" * 50)

#################################################################
## Randomized Search

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
print("=" * 50)
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print("=" * 50)

# Analyze the Best Models and Their Errors

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
print("%" * 35)
print("Feature Importances:", feature_importances)
print("%" * 35)

"""
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
"""


# ## Evaluate Your System on the Test Set

final_model = grid_search.best_estimator_


X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("=" * 50)
print("final_rmse:", final_rmse)
print("=" * 50)

#################################################################
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

# We could compute the interval manually like this:

m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# Alternatively, we could use a z-scores rather than t-scores:

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
