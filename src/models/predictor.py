import pathlib
import joblib


class Predictor:
    def __init__(self, path: pathlib.Path):
        self.model = joblib.load(path)

    def __call__(self, fps):
        scores = self.model.predict(fps)
        return scores
