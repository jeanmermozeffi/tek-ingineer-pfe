from typing import Callable, Dict, Optional, Tuple, Union, List
import numpy as np
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import Booster

# Importation de la fonction build_hyperparameters_space mise à jour
from src.utils.hyperparameters.shared import build_hyperparameters_space


def load_class(module_and_class_name: str) -> BaseEstimator:
    """
    module_and_class_name:
        ensemble.ExtraTreesRegressor
        ensemble.GradientBoostingRegressor
        ensemble.RandomForestRegressor
        linear_model.Lasso
        linear_model.LinearRegression
        linear_model.Ridge
        neighbors.KNeighborsRegressor
        tree.DecisionTreeRegressor
        svm.LinearSVR
        xgboost.Booster
        preprocessing.PolynomialFeatures
    """
    parts = module_and_class_name.split('.')
    cls = sklearn
    for part in parts:
        cls = getattr(cls, part)

    return cls


def train_model(
        model: BaseEstimator,
        X_train: csr_matrix,
        y_train: Series,
        X_val: Optional[csr_matrix] = None,
        eval_metric: Callable = root_mean_squared_error,
        fit_params: Optional[Dict] = None,
        y_val: Optional[Series] = None,
        **kwargs,
) -> Tuple[BaseEstimator, Optional[Dict], Optional[np.ndarray]]:
    model.fit(X_train, y_train, **(fit_params or {}))

    metrics = None
    y_pred = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)

        rmse = eval_metric(y_val, y_pred)
        mse = eval_metric(y_val, y_pred)
        metrics = dict(mse=mse, rmse=rmse)

    return model, metrics, y_pred


def get_integer_params(model_class: Callable[..., BaseEstimator]) -> List[str]:
    if model_class == LinearSVR:
        return ['max_iter']
    elif model_class == RandomForestRegressor:
        return ['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
    elif model_class == GradientBoostingRegressor:
        return ['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
    elif model_class == ExtraTreesRegressor:
        return ['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
    elif model_class == Lasso or model_class == Ridge:
        return ['max_iter']
    elif model_class == PolynomialFeatures:
        return ['degree']
    elif model_class == KNeighborsRegressor:
        return ['n_neighbors', 'leaf_size']
    elif model_class == DecisionTreeRegressor:
        return ['max_depth', 'min_samples_split', 'min_samples_leaf']
    elif model_class == SVR:
        return ['degree', 'max_iter']
    elif model_class == Booster:
        return ['num_boost_round', 'max_depth']
    return []


def tune_hyperparameters(
        model_class: Callable[..., BaseEstimator],
        X_train: csr_matrix,
        y_train: Series,
        X_val: csr_matrix,
        y_val: Series,
        callback: Optional[Callable[..., None]] = None,
        eval_metric: Callable[[Series, Series], float] = root_mean_squared_error,
        fit_params: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        max_evaluations: int = 50,
        random_state: int = 42,
) -> Dict:
    def __objective(
            params: Dict,
            X_train=X_train,
            X_val=X_val,
            callback=callback,
            eval_metric=eval_metric,
            fit_params=fit_params,
            model_class=model_class,
            y_train=y_train,
            y_val=y_val,
    ) -> Dict[str, Union[float, str]]:
        model, metrics, predictions = train_model(
            model_class(**params),
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            eval_metric=eval_metric,
            fit_params=fit_params,
        )

        if callback:
            callback(
                hyperparameters=params,
                metrics=metrics,
                model=model,
                predictions=predictions,
            )

        return dict(loss=metrics['rmse'], status=STATUS_OK)

    space, choices = build_hyperparameters_space(
        model_class,
        random_state=random_state,
        **(hyperparameters or {}),
    )

    best_hyperparameters = fmin(
        fn=__objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials(),
    )

    HYPERPARAMETERS_WITH_CHOICE_INDEX = list(choices.keys())

    # Convert choice index to choice value.
    for key in HYPERPARAMETERS_WITH_CHOICE_INDEX:
        if key in best_hyperparameters and key in choices:
            idx = int(best_hyperparameters[key])
            best_hyperparameters[key] = choices[key][idx]

    # for key in choices:
    #     if key in best_hyperparameters:
    #         idx = int(best_hyperparameters[key])
    #         best_hyperparameters[key] = choices[key][idx]

    # Convert float hyperparameters that should be integers
    integer_params = get_integer_params(model_class)
    for key in integer_params:
        if key in best_hyperparameters:
            best_hyperparameters[key] = int(best_hyperparameters[key])

    return best_hyperparameters


