import pandas as pd
from typing import List, Tuple

from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import seaborn as sns


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule et affiche la matrice de corrélation des caractéristiques numériques d'un DataFrame.

    Args:
    - df (DataFrame): Le DataFrame contenant les données.

    Returns:
    - DataFrame: La matrice de corrélation des caractéristiques numériques.
    """
    # Sélection des caractéristiques numériques
    numeric_features = df.select_dtypes(include=['number']).columns

    # Création du DataFrame avec les caractéristiques numériques
    df_numeric = df[numeric_features]

    # Calcul de la matrice de corrélation
    correlation_matrix = df_numeric.corr(method='pearson')

    # Affichage de la heatmap de la matrice de corrélation
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matrice de corrélation entre les caractéristiques numériques')
    plt.show()

    return correlation_matrix


def compute_p_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les p-values associées aux coefficients de corrélation de Pearson entre toutes les paires de colonnes numériques d'un DataFrame.

    Args:
    - df (DataFrame): Le DataFrame contenant les données.

    Returns:
    - DataFrame: Une matrice de p-values avec les p-values associées aux coefficients de corrélation.
    """

    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=['number'])

    columns = df_numeric.columns
    num_columns = len(columns)

    # Initialiser une matrice de p-values
    p_values = pd.DataFrame(index=columns, columns=columns)

    # Calculer les p-values pour chaque paire de colonnes
    for i in range(num_columns):
        for j in range(num_columns):
            if i != j:
                col1 = columns[i]
                col2 = columns[j]

                # Supprimer les lignes avec des NaN dans les colonnes concernées
                valid_data = df_numeric[[col1, col2]].dropna()

                if valid_data.shape[0] > 1:  # Assurer qu'il y a assez de données pour calculer la corrélation
                    # Calculer la corrélation et la p-value pour les colonnes
                    _, p_val = pearsonr(valid_data[col1], valid_data[col2])
                else:
                    p_val = None

                # Remplir la matrice de p-values
                p_values.loc[col1, col2] = p_val
            else:
                # Pour les mêmes colonnes, la p-value est non applicable
                p_values.loc[columns[i], columns[j]] = 1

    # Convertir les p-values en type float
    p_values = p_values.astype(float)

    return p_values


def find_significant_correlations(correlation_matrix: pd.DataFrame, p_values: pd.DataFrame) -> List[str]:
    """
    Trouve les colonnes ayant une corrélation significative avec la variable 'Prix' dans la matrice de corrélation.

    Args:
    - correlation_matrix (pd.DataFrame): La matrice de corrélation entre les caractéristiques.
    - p_values (pd.DataFrame): La matrice des p-values associées aux coefficients de corrélation.

    Returns:
    - List[str]: Les noms des colonnes ayant une corrélation significative avec 'Prix'.
    """
    # Assurer que la variable 'Prix' est dans la matrice de corrélation
    if 'Prix' not in correlation_matrix.columns:
        raise ValueError("La variable 'Prix' doit être présente dans la matrice de corrélation.")

    # Extraire la série de corrélations avec 'Prix'
    correlation_with_price = correlation_matrix['Prix']
    p_values_with_price = p_values['Prix']

    # Filtrer les colonnes basées sur les critères
    significant_columns = []
    for column in correlation_with_price.index:
        if column != 'Prix':  # Ignorer la colonne 'Prix' elle-même
            coef = correlation_with_price[column]
            p_val = p_values_with_price[column]
            if abs(coef) >= 0.30 and p_val < 0.05:
                significant_columns.append(column)

    return significant_columns
