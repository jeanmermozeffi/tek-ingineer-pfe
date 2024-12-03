import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from typing import List, Dict, Optional

from src.module.processing.transforme import convert_prices_to_cfa, explode_json_column
from src.utils.preprocessing.categorical_encoder import EncodeCategoricalFeatures, ApplyKNNImputer


drop_columns = {
    'Dimensions': [
        'Object_Folder_Dimensions',
        'coefficient_daerodynamisme',
        'hauteur_avec_barres_de_toit',
        'angle_dattaque',
        'angle_de_fuite',
        'angle_ventral'
    ],
    'Engine': ['Nom_du_moteur'],
    'Performance': ['Object_Folder_Performance', '0_a_1000_m_DA'],
    'Resumer': ['Object_Folder_Resumer', 'date_de_fin_de_commercialisation'],
    'Habitability': ['Object_Folder_Habitability', 'hauteur_de_seuil_de_chargement', 'longueur_utile',
                     'largeur_utile']
}


# cols_to_explode = ['Resumer', 'Dimensions', 'Weight', 'Habitability', 'Tires', 'Engine', 'Transmission', \
# 'Performance', 'Consumption']
#
# drop_cols_engine = ['Nom_du_moteur']
# drop_cols_habitability = ['Object_Folder_Habitability', 'hauteur_de_seuil_de_chargement', 'longueur_utile',
#                           'largeur_utile']
# drop_cols_performance = ['Object_Folder_Performance', '0_a_1000_m_DA']
# drop_cols_resume = ['Object_Folder_Resumer', 'date_de_fin_de_commercialisation']


def process_data(
        explose_column: str,
        drop_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        target_col: str = 'Prix',
        path: str = "../data/cleaning/liste_fiches_technical_details_cleaning.csv",
        output_dir: str = "../data/cleaning/relation",
        max_unique_categories: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Exécute l'ensemble du processus de transformation des données et renvoie chaque DataFrame intermédiaire.

    :param path: Chemin du fichier CSV contenant les données brutes.
    :param output_dir: Répertoire de sortie pour enregistrer les fichiers CSV transformés.
    :param explose_column: Colonne cible pour exploser
    :param drop_cols: Liste des colonnes à supprimer après explosion JSON.
    :param exclude_columns: Liste des colonnes à exclure de l'encodage.
    :param n_neighbors: Nombre de voisins pour KNNImputer.
    :param weights: Poids des voisins pour KNNImputer.
    :param target_col: Nom de la colonne cible.
    :param max_unique_categories: Nombre maximal de catégories uniques pour utiliser OneHotEncoder.
    :return: Un dictionnaire contenant les DataFrames intermédiaires.
    """
    # Lire le fichier CSV brut
    df_raw = pd.read_csv(path, sep=";")

    # Copier le DataFrame brut
    df = df_raw.copy()

    # Conversion des prix en CFA
    df = convert_prices_to_cfa(df)
    df['Prix'] = df['Prix'].astype(int)

    # Explosion de la colonne JSON
    df_result = explode_json_column(dataframe=df, column_name=explose_column)

    # Vérifier et Supprimer les colonnes spécifiées après explosion JSON
    if drop_cols is None:
        drop_cols = []
        if explose_column in drop_columns:
            drop_cols.extend(drop_columns[explose_column])
        else:
            drop_cols = [f"Object_Folder_{explose_column}"]

    if drop_cols is not None:
        existing_drop_cols = [col for col in drop_cols if col in df_result.columns]
        if existing_drop_cols:
            df_result.drop(columns=existing_drop_cols, inplace=True)

    # Vérifier et ajouter les colonnes manquantes
    if target_col not in df_result.columns:
        df_result[target_col] = df[target_col]

    if 'Immatriculation' not in df_result.columns:
        df_result['Immatriculation'] = df['Immatriculation']

    # Enregistrer le DataFrame après explosion JSON
    consumption_path = f"{output_dir}/{explose_column.lower()}_data.csv"
    df_result.to_csv(consumption_path, index=False, header=True)

    # Définir les colonnes à exclure si non spécifiées
    if exclude_columns is None:
        exclude_columns = ['Immatriculation']

    # Définir le pipeline
    pipeline = Pipeline([
        ('encode_categorical', EncodeCategoricalFeatures(
            target_col=target_col,
            exclude_cols=exclude_columns,
            max_unique_categories=max_unique_categories
        )),
        ('impute_missing', ApplyKNNImputer(
            n_neighbors=n_neighbors,
            weights=weights))
    ])

    # Appliquer le pipeline de transformations
    df_transformed = pipeline.fit_transform(df_result)
    df_transformed['Immatriculation'] = df['Immatriculation']

    # Enregistrer le DataFrame transformé
    transformed_path = f"{output_dir}/transformed/transformed_{explose_column.lower()}_data.csv"
    df_transformed.to_csv(transformed_path, index=False, header=True)

    # Retourner les DataFrames intermédiaires dans un dictionnaire
    return {
        'df_json_exploded': df_result,
        'df_transformed': df_transformed
    }



