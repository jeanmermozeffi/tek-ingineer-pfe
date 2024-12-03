import mlflow


def get_latest_model_version(model_name: str) -> int:
    """
    Obtient le numéro de la dernière version du modèle enregistré.

    :param model_name: Nom du modèle dont vous voulez obtenir la dernière version.
    :return: Numéro de la dernière version du modèle.
    """
    client = mlflow.MlflowClient()

    # Obtenir toutes les versions du modèle
    model_versions = client.list_registered_model_versions(model_name)

    if not model_versions:
        raise ValueError(f"Le modèle '{model_name}' n'existe pas ou n'a pas de versions enregistrées.")

    # Extraire les numéros de version et trouver le plus élevé
    versions = [int(version.version) for version in model_versions]
    latest_version = max(versions)

    return latest_version

# # Exemple d'utilisation
# model_name = "LinearRegression"
# latest_version = get_latest_model_version(model_name)
# print(f"La dernière version du modèle '{model_name}' est la version {latest_version}.")
