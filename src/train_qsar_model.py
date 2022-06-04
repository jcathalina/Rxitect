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
"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran 
"""
import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="qsar_train_config")
def train_model(config: DictConfig):
    """Function to train the model"""

    input_path = abspath(config.processed.path)
    output_path = abspath(config.final.path)

    print(f"Train modeling using {input_path}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {output_path}")


if __name__ == "__main__":
    train_model()
