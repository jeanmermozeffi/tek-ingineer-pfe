from typing import Tuple, Optional, List
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.impute import KNNImputer


class EncodeCategoricalFeatures(BaseEstimator, TransformerMixin):
    """
    Encode les variables catégorielles en utilisant OneHotEncoder pour les colonnes avec un nombre
    de catégories uniques inférieur ou égal à max_unique_categories, et TargetEncoder pour les autres colonnes.

    :param target_col: Le nom de la colonne cible (par défaut 'Prix').
    :param excluded_col: Le nom de la colonne à exclure de l'encodage (par défaut 'Immatriculation').
    :param max_unique_categories: Le nombre maximal de catégories uniques pour utiliser OneHotEncoder (par défaut 10).
    """

    def __init__(self, target_col: str = 'Prix', exclude_cols=None, max_unique_categories: int = 10):
        if exclude_cols is None:
            exclude_cols = ['Immatriculation']

        self.target_col = target_col
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.max_unique_categories = max_unique_categories
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.target_encoder = ce.TargetEncoder()
        self.category_cols = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EncodeCategoricalFeatures':
        """
       Identifie les colonnes catégorielles dans le DataFrame et prépare les encodeurs.

       :param X: DataFrame contenant les données à encoder.
       :param y: Série des valeurs cibles (facultatif pour certaines tâches).
       :return: L'instance actuelle de EncodeCategoricalFeatures.
       """
        if not isinstance(self.target_col, str):
            raise TypeError(f"Le nom de la colonne cible doit être une chaîne de caractères, mais a trouvé \
            {type(self.target_col)}")

        self.category_cols = X.select_dtypes(include=['object']).columns

        if self.target_col in self.category_cols:
            self.category_cols = self.category_cols.difference([self.target_col])
            # self.category_cols = self.category_cols.drop(self.target_col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l'encodage des variables catégorielles sur le DataFrame.

        :param X: DataFrame contenant les données à encoder.
        :return: DataFrame avec les variables catégorielles encodées.
        """

        df = X.copy()
        df_onehot_encoded = pd.DataFrame()
        df_target_encoded = pd.DataFrame()

        for col in self.category_cols:
            if col not in self.exclude_cols:
                num_unique = len(df[col].unique())

                if num_unique <= self.max_unique_categories:
                    # Appliquer OneHotEncoder
                    encoded = self.onehot_encoder.fit_transform(df[[col]])
                    encoded_df = pd.DataFrame(encoded, columns=self.onehot_encoder.get_feature_names_out([col]))
                    df_onehot_encoded = pd.concat([df_onehot_encoded, encoded_df], axis=1)
                else:
                    # Appliquer TargetEncoder
                    if self.target_col not in df.columns:
                        raise KeyError(f"La colonne cible '{self.target_col}' est manquante dans le DataFrame.")

                    encoded = self.target_encoder.fit_transform(df[col], df[self.target_col])
                    df_target_encoded[col] = encoded

        # Supprimer les colonnes originales catégorielles après l'encodage
        df = df.drop(columns=self.category_cols)
        # Fusionner les données encodées avec le DataFrame d'origine
        df = pd.concat([df, df_onehot_encoded, df_target_encoded], axis=1)

        return df


class ApplyKNNImputer(BaseEstimator, TransformerMixin):
    """
    Applique le KNNImputer pour imputer les valeurs manquantes dans un DataFrame en utilisant
    l'algorithme des k plus proches voisins.

    :param n_neighbors: Nombre de voisins à utiliser pour l'imputation (par défaut 5).
    :param weights: Poids des voisins utilisés pour l'imputation (par défaut 'uniform'). Les options sont 'uniform'\
     ou 'distance'.
    :param metric: Métrique de distance utilisée pour les voisins (par défaut 'nan_euclidean'). Les options incluent\
     'euclidean', 'manhattan', etc.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', metric: str = 'nan_euclidean'):
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ApplyKNNImputer':
        """
        Apprend les paramètres d'imputation à partir des données fournies.

        :param X: DataFrame contenant les données avec des valeurs manquantes à imputer.
        :param y: Optionnel. Série des valeurs cibles, non utilisée pour l'imputation mais nécessaire \
        pour la compatibilité avec `TransformerMixin`.
        :return: Instance de la classe `ApplyKNNImputer`.
        """
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute les valeurs manquantes dans le DataFrame en utilisant les paramètres appris.

        :param X: DataFrame contenant les données avec des valeurs manquantes à imputer.
        :return: DataFrame avec les valeurs manquantes imputées.
        """
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, columns=X.columns)


def encode_categorical_features(
        df: pd.DataFrame,
        target_col: str = 'Prix',
        excluded_col: str = 'Immatriculation',
        max_unique_categories: int = 10
) -> pd.DataFrame:
    """
    Encode les variables catégorielles en utilisant OneHotEncoder pour les colonnes avec un nombre
    de catégories uniques inférieur ou égal à max_unique_categories, et TargetEncoder pour les autres colonnes.

    :param df: DataFrame contenant les données à encoder.
    :param target_col: Le nom de la colonne cible (Prix).
    :param excluded_col: Le nom de la colonne à exclure de l'encodage (par défaut 'Immatriculation').
    :param max_unique_categories: Le nombre maximal de catégories uniques pour utiliser OneHotEncoder.
    :return: DataFrame avec les variables catégorielles encodées.
    """
    # Sélectionner les colonnes catégorielles
    category_cols = df.select_dtypes(include=['object']).columns

    # Initialiser les encoders
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    target_encoder = ce.TargetEncoder()

    # Créer des DataFrames pour stocker les résultats
    df_onehot_encoded = pd.DataFrame()
    df_target_encoded = pd.DataFrame()

    # Appliquer l'encodage selon le nombre de catégories uniques
    for col in category_cols:
        if col != excluded_col:  # Exclure la colonne spécifiée
            num_unique = len(df[col].unique())

            if num_unique <= max_unique_categories:
                # Appliquer OneHotEncoder
                encoded = onehot_encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out([col]))
                df_onehot_encoded = pd.concat([df_onehot_encoded, encoded_df], axis=1)
            else:
                # Appliquer TargetEncoder
                encoded = target_encoder.fit_transform(df[col], df[target_col])
                df_target_encoded[col] = encoded

    # Supprimer les colonnes originales catégorielles après l'encodage
    df_encoded = df.drop(columns=category_cols)

    # Fusionner les données encodées avec le DataFrame d'origine
    df_encoded = pd.concat([df_encoded, df_onehot_encoded, df_target_encoded], axis=1)

    return df_encoded


def apply_knn_imputer(
        df: pd.DataFrame,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'nan_euclidean',
) -> pd.DataFrame:
    """
    Applique l'imputation KNN pour remplir les valeurs manquantes dans un DataFrame.

    :param df: DataFrame avec les valeurs manquantes à imputer.
    :param n_neighbors: Nombre de voisins à utiliser pour l'imputation.
    :param weights: Poids à appliquer lors de l'imputation ('uniform' ou 'distance').
    :param metric: Métrique à utiliser pour la distance des voisins.

    :return: DataFrame avec les valeurs manquantes imputées.
    """
    # Initialiser l'imputeur KNN
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )

    # Appliquer l'imputation
    imputed_array = imputer.fit_transform(df)

    # Convertir le tableau numpy résultant en DataFrame avec les mêmes colonnes
    df_imputed = pd.DataFrame(imputed_array, columns=df.columns)

    return df_imputed
