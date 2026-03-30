import os
import pickle

import numpy as np
import pytest
from sklearn.metrics import f1_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TEST_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "test_sample")

MODEL_NAMES = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "CategoricalNB",
    "SVC",
]

MIN_F1_SCORE = 0.35


@pytest.fixture(scope="module")
def test_data():
    with open(os.path.join(TEST_SAMPLE_DIR, "X_test_scaled.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(TEST_SAMPLE_DIR, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    y_true = np.array(y_test).reshape(-1)
    return X_test, y_true


@pytest.fixture(scope="module")
def loaded_models():
    models = {}
    for name in MODEL_NAMES:
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
            models[name] = pickle.load(f)
    return models


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_f1_score(model_name, loaded_models, test_data):
    X_test, y_true = test_data
    model = loaded_models[model_name]
    y_pred = model.predict(X_test)
    score = f1_score(y_true, y_pred, average="micro")
    assert score > MIN_F1_SCORE, (
        f"{model_name}: F1 micro = {score:.4f}, esperado > {MIN_F1_SCORE}"
    )
