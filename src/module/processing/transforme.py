import ast
import re
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

cols_convert_int = [
    'Prix',
    'volume_de_coffre_utile',
    'poids_a_vide',
    'volume_de_coffre',
    'poids_a_vide',
    'ptac',
    'ptra',
    'charge_utile',
    'poids_tracte_freine',
    'poids_tracte_non_freine',
    'nombre_de_places',
    'Vitesse_maximale',
    'Emission_de_CO2',
    'Course',
    'Alesage',
    'Puissance_reelle_maxi_kW',
    'Puissance_reelle_maxi_ch',
    'Nombre_de_soupapes',
    'Couple_maxi',
    'Au_regime_de',
    'Cylindree',
    'angle_ventral',
    'angle_dattaque',
    'angle_de_fuite',
    'garde_au_sol',
    'reservoir',
]

cols_drop = [
    'date_de_fin_de_commercialisation',
    'Date Publication',
    'Immatriculation',
    'Nom_du_moteur',
    'emission_de_co2',
    'puissance_commerciale',
    'angle_de_fuite',
    'angle_ventral',
    'angle_dattaque',
    'garde_au_sol',
    'Vehicule',
    'energie',
    'boite_de_vitesses',
    'Architecture',
    'Alimentation',
    'Disposition_du_moteur',
    'Mixte',
    'Cycle_urbain',
    'Extra_urbain',
    'types_de_pneumatiques',
    'taille_des_roues_avant',
    'taille_des_roues_arriere',
    'type_de_roues_de_secours',
    'puissance_fiscale',
    'consommation_mixte',
    'carrosserie',
    '0_a_100_km/h',
    'Mode_de_transmission',
    'volume_de_coffre_utile',
    'poids_tracte_freine',
    'poids_tracte_non_freine',
    'Emission_de_CO2',
    'Alesage',
    'Course',
    'Norme_anti-pollution',
    'Nombre_de_soupapes',
    'Couple_maxi',
    'Injection',
    'porte_a_faux_arriere',
    'porte_a_faux_avant',
    'Au_regime_de'
]


def clean_numeric_string(value):
    """
    Supprime tous les caractères non numériques d'une chaîne de caractères.
    Si la chaîne devient vide après le nettoyage, retourne NaN.
    """
    # Supprimer tous les caractères non numériques, sauf les points (pour les flottants)
    cleaned_value = re.sub(r'[^\d.]+', '', str(value))

    # Si après le nettoyage la chaîne est vide, retourner NaN
    return float(cleaned_value) if cleaned_value else float('nan')


def convert_columns_to_int(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_string)
            df[col] = df[col].fillna(0).astype(float).astype(int)
    return df


def convert_prices_to_cfa(dataframe: pd.DataFrame, col_name: str = 'Prix', parity: float = 667.66):
    """
    Convertit les prix d'une colonne donnée en CFA en utilisant un taux de conversion spécifique.

    :param dataframe: Le DataFrame à modifier.
    :param col_name: Le nom de la colonne à convertir. Par défaut, 'Prix'.
    :param parity: Le taux de conversion Euro vers CFA. Par défaut, 655.957.
    :return: Le DataFrame modifié avec les prix convertis.
    """
    if col_name in dataframe.columns:
        # Convertir les prix en CFA et arrondir à l'entier le plus proche
        dataframe[col_name] = (dataframe[col_name] * parity).round().astype(int)
    return dataframe


def one_hot_encode(dataframe: pd.DataFrame, cols: list):
    """
    Encodes specified categorical columns in the DataFrame using one-hot encoding.

    Args:
        dataframe (pd.DataFrame): The DataFrame to encode.
        cols (list): List of column names to encode.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    df_encoded = pd.get_dummies(dataframe, columns=cols, drop_first=True)
    return df_encoded


def label_encode(dataframe: pd.DataFrame, cols: list):
    """
    Encodes specified categorical columns in the DataFrame using label encoding.

    Args:
        dataframe (pd.DataFrame): The DataFrame to encode.
        cols (list): List of column names to encode.

    Returns:
        pd.DataFrame: The DataFrame with label encoded columns.
    """
    df_encoded = dataframe.copy()
    for col in cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def replace_missing_with_mean(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in dataframe.select_dtypes(include=['float', 'int']).columns:
        mean_value = round(dataframe[column].mean(), 2)
        dataframe[column] = dataframe[column].fillna(mean_value)
    return dataframe


def replace_missing_with_median(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in dataframe.select_dtypes(include=['float', 'int']).columns:
        mean_value = round(dataframe[column].median(), 2)
        dataframe[column] = dataframe[column].fillna(mean_value)
    return dataframe


def replace_missing_with_mode(dataframe: pd.DataFrame) -> pd.DataFrame:
    for column in dataframe.select_dtypes(include=['object']).columns:
        mode_value = dataframe[column].mode()[0]  # [0] pour récupérer la première valeur en cas d'égalité
        dataframe[column] = dataframe[column].fillna(mode_value)
    return dataframe


def explode_json_column(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    dataframe[column_name] = dataframe[column_name].apply(ast.literal_eval)
    exploded_df = pd.json_normalize(dataframe[column_name])
    # Renommer les colonnes si nécessaire (ici, on garde les noms originaux)
    exploded_df.columns = [f"{col}" for col in exploded_df.columns]

    return exploded_df


def create_dfs_from_csvs(directory_path):
    dfs = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            dataframe = pd.read_csv(file_path)
            # Ajouter le DataFrame au dictionnaire avec le nom du fichier (sans extension) comme clé
            dfs[os.path.splitext(filename)[0]] = dataframe

    return dfs
