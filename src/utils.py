import os
import sys

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_obj(file_path, obj):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[list(models.keys()).index(model_name)]
            para = param[model_name]

            gv = GridSearchCV(model, para, cv=3)

            gv.fit(X_train, y_train)

            model.set_params(**gv.best_params_)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_score
        return report

    except Exception as e:
        raise CustomException(e)
