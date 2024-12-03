from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.base import Apply
from hyperopt.pyll import scope
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

import mlflow

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import Booster


def build_hyperparameters_space(
        model_class: Callable[
            ...,
            BaseEstimator
        ],
        random_state: int = 42,
        **kwargs,
) -> Tuple[Dict, Dict[str, List]]:
    params = {}
    choices = {}

    if LinearSVR is model_class:
        params = dict(
            epsilon=hp.uniform('epsilon', 0.0, 1.0),
            C=hp.loguniform('C', -7, 3),
            max_iter=scope.int(hp.quniform('max_iter', 1000, 5000, 100)),
        )

    if RandomForestRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform('max_depth', 5, 45, 5)),
            min_samples_leaf=scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            n_estimators=scope.int(hp.quniform('n_estimators', 10, 60, 10)),
            max_features=hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
            bootstrap=hp.choice('bootstrap', [True, False]),
            criterion=hp.choice('criterion', ['mse', 'mae']),
            max_samples=scope.int(hp.uniform('max_samples', 0.5, 1.0)),
            random_state=random_state,
        )

        choices['max_features'] = ['auto', 'sqrt', 'log2', None]
        choices['criterion'] = ['mse', 'mae']
        choices['bootstrap'] = [True, False]

    if GradientBoostingRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform('max_depth', 5, 40, 1)),
            min_samples_leaf=scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            n_estimators=scope.int(hp.quniform('n_estimators', 10, 50, 10)),
            random_state=random_state,
        )

    if ExtraTreesRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform('max_depth', 5, 30, 5)),
            min_samples_leaf=scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            min_samples_split=scope.int(hp.quniform('min_samples_split', 2, 20, 2)),
            n_estimators=scope.int(hp.quniform('n_estimators', 10, 40, 10)),
            random_state=random_state,
        )

    if Lasso is model_class or Ridge is model_class:
        params = dict(
            alpha=hp.uniform('alpha', 0.0001, 1.0),
            max_iter=scope.int(hp.quniform('max_iter', 1000, 5000, 100)),
        )

    if LinearRegression is model_class:
        choices['fit_intercept'] = [True, False]

    if PolynomialFeatures is model_class:
        params = dict(
            degree=scope.int(hp.quniform('degree', 2, 5, 1)),
            interaction_only=hp.choice('interaction_only', [True, False]),
            include_bias=hp.choice('include_bias', [True, False]),
        )

        choices['interaction_only'] = [True, False]
        choices['include_bias'] = [True, False]

    if KNeighborsRegressor is model_class:
        params = dict(
            n_neighbors=scope.int(hp.quniform('n_neighbors', 1, 50, 1)),
            weights=hp.choice('weights', ['uniform', 'distance']),
            algorithm=hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            leaf_size=scope.int(hp.quniform('leaf_size', 10, 50, 1)),
            p=hp.choice('p', [1, 2]),
        )

        choices['weights'] = ['uniform', 'distance']
        choices['algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
        choices['p'] = [1, 2]

    if DecisionTreeRegressor is model_class:
        params = dict(
            max_depth=scope.int(hp.quniform('max_depth', 1, 40, 1)),
            min_samples_split=scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            min_samples_leaf=scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
            random_state=random_state,
        )

    if Booster is model_class:
        params = dict(
            colsample_bytree=hp.uniform('colsample_bytree', 0.5, 1.0),
            gamma=hp.uniform('gamma', 0.1, 1.0),
            learning_rate=hp.loguniform('learning_rate', -3, 0),
            max_depth=scope.int(hp.quniform('max_depth', 4, 100, 1)),
            min_child_weight=hp.loguniform('min_child_weight', -1, 3),
            num_boost_round=hp.quniform('num_boost_round', 500, 1000, 10),
            objective='reg:squarederror',
            random_state=random_state,
            reg_alpha=hp.loguniform('reg_alpha', -5, -1),
            reg_lambda=hp.loguniform('reg_lambda', -6, -1),
            subsample=hp.uniform('subsample', 0.1, 1.0),
        )

    if SVR is model_class:
        params = dict(
            C=hp.loguniform('C', -3, 3),  # Paramètre de régularisation, de 0.001 à 20 environ
            epsilon=hp.loguniform('epsilon', -3, 0),  # Epsilon dans la fonction de perte du SVR, de 0.001 à 1
            kernel=hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),  # Choix du noyau
            degree=scope.int(hp.quniform('degree', 2, 5, 1)),  # Degré du polynôme pour les noyaux 'poly'
            gamma=hp.choice('gamma', ['scale', 'auto']),  # Coefficient pour les noyaux 'rbf', 'poly' et 'sigmoid'
            coef0=hp.uniform('coef0', 0, 1),  # Terme indépendant dans le noyau 'poly' et 'sigmoid'
            tol=hp.loguniform('tol', -5, -1),  # Tolérance pour le critère d'arrêt, de 0.00001 à 0.1
            max_iter=scope.int(hp.quniform('max_iter', 1000, 5000, 500))  # Nombre maximal d'itérations
        )

        choices['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
        choices['gamma'] = ['scale', 'auto']

    # # Automatiquement ajouter les choix dans le dictionnaire choices
    # for key, value in params.items():
    #     if isinstance(value, Apply) and value.name == 'switch':
    #         choices[key] = value.pos_args[1:]

    # Ajouter les choix manuels si spécifiés
    for key, value in choices.items():
        params[key] = hp.choice(key, value)

    if kwargs:
        for key, value in kwargs.items():
            if value is not None:
                kwargs[key] = value

    return params, choices


def map_hyperparameters(model_class: Callable, best_params: Dict[str, Union[int, float]], int_params=None) -> Dict[
    str, Union[int, float, str]]:
    """
    Mappe les hyperparamètres trouvés par Hyperopt aux valeurs appropriées pour l'entraînement du modèle.

    Args:
    - model_class: Le modèle à entraîner (ex: KNeighborsRegressor, LinearSVR, etc.).
    - best_params: Dictionnaire des hyperparamètres trouvés par Hyperopt.

    Returns:
    - Dictionnaire des hyperparamètres avec les valeurs mappées prêtes à être utilisées pour entraîner le modèle.
    """
    # Dictionnaires de choix pour chaque modèle
    if int_params is None:
        int_params = {}

    mapping_choices = {
        KNeighborsRegressor: {
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        },
        PolynomialFeatures: {
            'interaction_only': [True, False],
            'include_bias': [True, False]
        },
        ExtraTreesRegressor: {
            'criterion': ['mse', 'mae'],
        },
        RandomForestRegressor: {
            'criterion': ['mse', 'mae'],
        },
        GradientBoostingRegressor: {
            'loss': ['ls', 'lad', 'huber', 'quantile'],
            'criterion': ['friedman_mse', 'mse', 'mae'],
        },
        DecisionTreeRegressor: {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random']
        },
        LinearSVR: {
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        },
        Lasso: {
            'selection': ['cyclic', 'random'],
        },
        Ridge: {
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        },
        SVR: {
            'kernel', ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma', ['scale', 'auto']
        },

    }

    # Si le modèle a des choix mappés, nous les appliquons
    if model_class in mapping_choices:
        for param, choices in mapping_choices[model_class].items():
            if param in best_params:
                best_params[param] = choices[int(best_params[param])]

    # Conversion des paramètres en entiers si nécessaire
    if int_params:
        for param in int_params:
            if param in best_params:
                best_params[param] = int(best_params[param])

    return best_params


def evaluate_model(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        # error_mean_square: List[float],
        # error_mean_absolute: List[float]
) -> Tuple[float, float]:
    """
    Évalue les performances du modèle en calculant les métriques d'erreur et le coefficient de détermination (R²).

    :param y_pred: Prédictions du modèle sur les données de test.
    :param y_true: Valeurs réelles des données de test.
    :param model: Modèle utilisé pour les prédictions.
    :param X_train: Données d'entraînement pour le calcul du R² sur l'entraînement.
    :param y_train: Valeurs réelles d'entraînement pour le calcul du R² sur l'entraînement.
    :param X_test: Données de test pour le calcul du R² sur les tests.
    :param y_test: Valeurs réelles de test pour le calcul du R² sur les tests.
    :param error_mean_square: Liste pour stocker les valeurs du RMSE.
    :param error_mean_absolute: Liste pour stocker les valeurs du MAE.
    :return: Tuple contenant le RMSE et le MAE.
    """
    # Calcul des métriques d'erreur
    rmse = mean_squared_error(y_true, y_pred, squared=False)  # Calcul du RMSE
    mae = mean_absolute_error(y_true, y_pred)  # Calcul du MAE

    # Calcul et affichage du coefficient de détermination (R²)
    r2_train = model.score(X_train, y_train) * 100
    r2_test = model.score(X_test, y_test) * 100

    # Affichage des résultats
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}\n")
    print("R² for train is:", r2_train)
    print("R² for test is:", r2_test)

    # Journaliser les résultats dans MLflow
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R² Train", r2_train)
    mlflow.log_metric("R² Test", r2_test)

    return rmse, mae


def format_and_combine_predictions(
        y_predict: Union[list, pd.Series],
        y_test: pd.Series
) -> pd.DataFrame:
    """
    Formate les prédictions et les combine avec les valeurs réelles en un DataFrame.

    :param y_predict: Liste ou Series contenant les valeurs prédites par le modèle.
    :param y_test: Series contenant les valeurs réelles pour les données de test.
    :return: DataFrame contenant les prédictions et les valeurs réelles.
    """
    # Formater les prédictions en chaînes de caractères
    y_predict = [format(pred, 'f') for pred in y_predict]

    # Convertir les prédictions en DataFrame
    y_predict_df = pd.DataFrame(y_predict, columns=['Predicted'])
    y_test_df = pd.DataFrame(y_test, columns=['RealPrice'])
    results = pd.concat([y_predict_df, y_test_df.reset_index(drop=True)], axis=1)

    return results





