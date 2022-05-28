"""
# Example W&B Tracking for sklearn
wandb.init(project="my-test-project", entity="rxitect")


# Load data
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X, y = X[::2], y[::2]  # subsample for faster demo
wandb.errors.term._show_warnings = False
# ignore warnings about charts being built from subset of data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model, get predictions
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')

wandb.finish()
"""
