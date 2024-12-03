from typing import Union
import os
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from sklearn.base import BaseEstimator

from src.utils.hyperparameters.shared import format_and_combine_predictions
from src.utils.models import sklearn


def setup_mlflow_experiment(experiment_name: str) -> None:
    """
    Crée ou définit un nom d'expérience MLflow.

    :param experiment_name: Nom de l'expérience MLflow à créer ou à définir.
    """
    try:
        # Tentative de création de l'expérience
        mlflow.create_experiment(experiment_name)
    except MlflowException as e:
        # L'expérience existe déjà, ou une autre erreur est survenue
        print(f"Erreur lors de la création de l'expérience : {e}")

    # Définir l'expérience active
    mlflow.set_experiment(experiment_name)
    print(f"Expérience MLflow définie sur : {experiment_name}")


def log_experiment_results(
        experiment_name: str,
        sk_model: BaseEstimator,
        run_id: str,
        model_name: str,
        X_train: pd.DataFrame,
        y_predict: Union[list, pd.Series],
        y_test: pd.Series,
        artifact_path: str = "results",
        is_save_mdodel: bool = False,
) -> pd.DataFrame:
    """
    Enregistre les résultats de l'expérience dans MLflow.

    :param is_save_mdodel: True Sauvegarde du model sinon False
    :param X_train: données d'entrainement
    :param sk_model: Model
    :param run_id: Id de l'entrainement
    :param experiment_name: Nom de l'expérience MLflow.
    :param model_name: Nom du modèle utilisé.
    :param y_predict: Liste ou Series contenant les valeurs prédites par le modèle.
    :param y_test: Series contenant les valeurs réelles pour les données de test.
    :param params: Dictionnaire des paramètres du modèle.
    :param metrics: Dictionnaire des métriques du modèle.
    :param artifact_path: Chemin pour enregistrer les fichiers de résultats.
    """
    # Créer ou définir l'expérience MLflow
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(experiment_name)

    # Créer le répertoire si nécessaire
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)

    # Log des métriques du modèle
    # for key, value in metrics.items():
    #     mlflow.log_metric(key, value)

    # Format et combiner les prédictions et les valeurs réelles
    results_df = format_and_combine_predictions(y_predict, y_test)

    # Enregistrer le DataFrame des résultats en tant que fichier CSV
    results_file_path = f"{artifact_path}/results.csv"
    results_df.to_csv(results_file_path, index=False)
    mlflow.log_artifact(results_file_path)

    # Enregistrer le modèle dans MLflow
    input_example = X_train[:1]

    if is_save_mdodel:
        mlflow.sklearn.log_model(
            sk_model=sk_model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=input_example,
            signature=mlflow.models.infer_signature(X_train, y_predict)
        )

        print(f"Les résultats de l'expérience sont enregistrés dans l'expérience '{experiment_name}' \
        sous l'ID de run '{run_id}'.")

    return results_df

