import ast
from datetime import datetime

import pandas as pd


def clean_prix_column(df):
    # Retirer le signe euro, enlever les espaces à l'intérieur des nombres, et convertir en float
    df['Prix'] = df['Prix'].replace({'€': '', ' ': ''}, regex=True).astype(float)
    return df


def convert_to_date(df):
    def format_date(date_str):
        try:
            # Convertir la date en ajoutant le jour 01 pour compléter la date
            return pd.to_datetime('01-' + date_str, format='%d-%m-%Y', errors='coerce')
        except ValueError:
            return pd.NaT  # Retourner NaT (Not a Time) en cas d'erreur

    # Appliquer la fonction à la colonne spécifiée
    df['Date Publication'] = df['Date Publication'].apply(format_date)
    return df


def replace_single_with_double_quotes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        # Vérifier si la colonne est de type string
        if df[col].dtype == 'object':
            # Remplacer les apostrophes par des guillemets doubles
            df[col] = df[col].apply(lambda x: x.replace("'", "\"") if isinstance(x, str) else x)
    return df


def convert_column_to_int(df, column_name):
    """
    Convertit une colonne spécifiée d'un DataFrame en type int.

    :param df: DataFrame contenant la colonne à convertir.
    :param column_name: Nom de la colonne à convertir.
    :return: DataFrame avec la colonne convertie en type int.
    """
    # Assurer que la colonne existe dans le DataFrame
    if column_name in df.columns:
        # Convertir la colonne en int après avoir supprimé les valeurs manquantes
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
    else:
        print(f"Erreur : La colonne '{column_name}' n'existe pas dans le DataFrame.")
    return df


def clean_resumer(resumer_str):
    # Convert the string to a dictionary
    resumer_dict = ast.literal_eval(resumer_str)

    # Function to remove units and convert to float or int
    def clean_value(value, unit=None):
        # Handle non-numeric or missing values
        if value in ['nc', 'na', '-', '']:
            return None
        if unit:
            value = value.replace(unit, '').replace(',', '.').replace(' ', '')
        return float(value) if '.' in value else int(value)

    # Convert 'puissance_commerciale' from 'ch' to integer
    if 'puissance_commerciale' in resumer_dict:
        resumer_dict['puissance_commerciale'] = clean_value(resumer_dict['puissance_commerciale'], 'ch')

    # Convert 'puissance_fiscale' from 'CV' to integer
    if 'puissance_fiscale' in resumer_dict:
        resumer_dict['puissance_fiscale'] = clean_value(resumer_dict['puissance_fiscale'], 'CV')

    # Convert 'consommation_mixte' from 'L/100 Km' to float
    if 'consommation_mixte' in resumer_dict:
        resumer_dict['consommation_mixte'] = clean_value(resumer_dict['consommation_mixte'], 'L/100 Km')

    # Convert 'emission_de_co2' from 'g/km' to integer
    if 'emission_de_co2' in resumer_dict:
        co2_value = resumer_dict['emission_de_co2'].split(' ')[0]
        resumer_dict['emission_de_co2'] = clean_value(co2_value, 'g/km')

    return resumer_dict


def clean_consumption(consumption_str):
    # Convert the string to a dictionary
    consumption_dict = ast.literal_eval(consumption_str)

    # Function to remove units and convert to float
    def clean_value(value):
        # Remove the units
        value = value.replace(' L/100km', '').replace(' g/km', '').replace(',', '.')
        # Convert to float or int
        return float(value) if '.' in value else int(value)

    # Apply the cleaning function to each relevant key
    for key in ['Cycle_urbain', 'Extra_urbain', 'Mixte', 'Emission_de_CO2']:
        if key in consumption_dict:
            consumption_dict[key] = clean_value(consumption_dict[key])

    return consumption_dict


def clean_performance(performance_str):
    # Convert the string to a dictionary
    performance_dict = ast.literal_eval(performance_str)

    # Function to remove units and convert to float or int
    def clean_value(value):
        value = value.replace(' km/h', '').replace(' s', '').replace(',', '.')
        return float(value) if '.' in value else int(value)

    # Apply the cleaning function to each relevant key
    for key in ['Vitesse_maximale', '0_a_100_km/h']:
        if key in performance_dict:
            performance_dict[key] = clean_value(performance_dict[key])

    return performance_dict


def clean_technical(technical_str):
    # Convert the string to a list of dictionaries
    technical_list = ast.literal_eval(technical_str)

    # Function to clean and convert numeric values
    def clean_value(value):
        # Check if the value is numeric with a unit
        if ' m' in value:
            value = value.replace(' m', '').replace(',', '.')
            return float(value)
        return value  # Return the value as is if it's not numeric

    # Apply the cleaning function to each relevant key
    for item in technical_list:
        if 'Diametre_de_braquage_(trottoir)' in item:
            item['Diametre_de_braquage_(trottoir)'] = clean_value(item['Diametre_de_braquage_(trottoir)'])

    return technical_list


def clean_habitability(habitability_str):
    # Convert the string to a dictionary
    habitability_dict = ast.literal_eval(habitability_str)

    # Function to remove units and convert to float or int
    def clean_value(value):
        value = value.replace(' l', '').replace(' mm', '').replace(' ', '').replace(',', '.')
        return float(value) if '.' in value else int(value)

    # Apply the cleaning function to each relevant key
    for key in ['volume_de_coffre', 'volume_de_coffre_utile', 'hauteur_de_seuil_de_chargement', 'longueur_utile',
                'largeur_utile']:
        if key in habitability_dict:
            habitability_dict[key] = clean_value(habitability_dict[key])

    # Convert 'nombre_de_places' to int
    if 'nombre_de_places' in habitability_dict:
        habitability_dict['nombre_de_places'] = int(habitability_dict['nombre_de_places'])

    return habitability_dict


def clean_dimensions(dimensions_str):
    # Convert the string to a dictionary
    dimensions_dict = ast.literal_eval(dimensions_str)

    # Function to remove units and convert to float or int
    def clean_value(value):
        value = value.strip()
        value = value.replace('m', '').replace('l', '').replace('mm', '').replace('°', '').replace(',', '.')
        return float(value) if '.' in value else int(value)

    # Apply the cleaning function to each relevant key
    keys_to_clean = ['longueur', 'largeur', 'hauteur', 'empattement', 'reservoir',
                     'porte_a_faux_avant', 'porte_a_faux_arriere', 'voies_avant',
                     'voies_arriere', 'garde_au_sol', 'angle_dattaque', 'angle_ventral',
                     'angle_de_fuite']

    for key in keys_to_clean:
        if key in dimensions_dict:
            dimensions_dict[key] = clean_value(dimensions_dict[key])

    return dimensions_dict


def clean_weight(weight_str):
    # Convert the string to a dictionary
    weight_dict = ast.literal_eval(weight_str)

    # Function to remove units and convert to float or int
    def clean_value(value):
        value = value.strip()
        value = value.replace('kg', '').replace(',', '.').replace(' ', '')
        return int(value)

    # Apply the cleaning function to each relevant key
    keys_to_clean = ['poids_a_vide', 'ptac', 'ptra', 'charge_utile',
                     'poids_tracte_freine', 'poids_tracte_non_freine']

    for key in keys_to_clean:
        if key in weight_dict:
            weight_dict[key] = clean_value(weight_dict[key])

    return weight_dict


def clean_engine(engine_str):
    # Convert the string to a list of dictionaries
    try:
        engine_list = ast.literal_eval(engine_str)
    except (ValueError, SyntaxError):
        return None  # Return None if the string cannot be converted

    # Function to remove units, commas, and spaces, and convert to float or int
    def clean_value(value, unit=None):
        if value in ['nc', '-', '']:
            return None
        if unit:
            value = value.replace(unit, '').replace(',', '.').replace(' ', '')
        value = value.replace(',', '.').replace(' ', '')
        try:
            return float(value)
        except ValueError:
            return value  # Return the original value if it can't be converted

    for engine in engine_list:
        # Convert 'Cylindree' from 'cm³' to float
        if 'Cylindree' in engine:
            engine['Cylindree'] = clean_value(engine['Cylindree'], 'cm³')

        if 'Nom_du_moteur' in engine:
            engine['Nom_du_moteur'] = clean_value(engine['Nom_du_moteur'], 'T')

        # Convert 'Puissance_reelle_maxi' from 'ch' and 'kW' to float
        # Convert 'Puissance_reelle_maxi' from 'ch' and 'kW' to separate columns
        if 'Puissance_reelle_maxi' in engine:
            puissance = engine['Puissance_reelle_maxi'].split('/')
            if len(puissance) == 2:  # Ensure there are two parts
                engine['Puissance_reelle_maxi_ch'] = clean_value(puissance[0], 'ch')
                engine['Puissance_reelle_maxi_kW'] = clean_value(puissance[1], 'kW')
            else:
                engine['Puissance_reelle_maxi_ch'] = clean_value(puissance[0], 'ch')
                engine['Puissance_reelle_maxi_kW'] = None  # Handle cases where 'kW' is missing
            del engine['Puissance_reelle_maxi']  # Remove the original column

        # Convert 'Au_regime_de' from 'tr/min' to float
        if 'Au_regime_de' in engine:
            engine['Au_regime_de'] = clean_value(engine['Au_regime_de'], 'tr/min')

        # Convert 'Couple_maxi' from 'Nm' to float
        if 'Couple_maxi' in engine:
            engine['Couple_maxi'] = clean_value(engine['Couple_maxi'], 'Nm')

        # Split 'Alesage/course' and convert to float
        if 'Alesage/course' in engine:
            parts = engine['Alesage/course'].split('x')
            if len(parts) == 2:  # Ensure there are two parts
                alesage, course = parts
                engine['Alesage'] = clean_value(alesage.strip())
                engine['Course'] = clean_value(course.strip())
            else:
                engine['Alesage'] = None
                engine['Course'] = None
            del engine['Alesage/course']

        # Convert 'Rapport_volumetrique' to float
        if 'Rapport_volumetrique' in engine:
            engine['Rapport_volumetrique'] = clean_value(engine['Rapport_volumetrique'].split(':')[0].strip())

        if 'Nombre_de_soupapes' in engine:
            engine['Nombre_de_soupapes'] = clean_value(engine['Nombre_de_soupapes'].strip())

    return engine_list


def clean_tires(tires_str):
    # Convertir la chaîne en dictionnaire
    try:
        tires_dict = ast.literal_eval(tires_str)
    except (ValueError, SyntaxError):
        return None  # Retourner None si la chaîne ne peut pas être convertie

    # Nettoyer les valeurs
    def clean_value(value):
        if value in ['nc', '-', '']:
            return None
        return value.strip()

    # Séparer les tailles des roues avant et arrière
    if 'taille_des_roues_avant' in tires_dict:
        tires_dict['taille_des_roues_avant'] = clean_value(tires_dict['taille_des_roues_avant'])

    if 'taille_des_roues_arriere' in tires_dict:
        tires_dict['taille_des_roues_arriere'] = clean_value(tires_dict['taille_des_roues_arriere'])

    return tires_dict


def cleaning(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Application des transformations avec pipe
    df_clean = (dataframe
                .assign(Consumption_clean=lambda x: x['Consumption'].apply(clean_consumption))
                .assign(Performance_clean=lambda x: x['Performance'].apply(clean_performance))
                .assign(Technical_clean=lambda x: x['Technical'].apply(clean_technical))
                .assign(Habitability_clean=lambda x: x['Habitability'].apply(clean_habitability))
                .assign(Dimensions_clean=lambda x: x['Dimensions'].apply(clean_dimensions))
                .assign(Weight_clean=lambda x: x['Weight'].apply(clean_weight))
                .assign(Engine_clean=lambda x: x['Engine'].apply(clean_engine))
                .assign(Resumer_clean=lambda x: x['Resumer'].apply(clean_resumer))
                .assign(Tires_clean=lambda x: x['Tires'].apply(clean_tires))
                .pipe(lambda df: df.drop(columns=['Consumption', 'Performance', 'Technical', 'Habitability',
                                                  'Dimensions', 'Weight', 'Engine', 'Resumer', 'Tires']))
                .pipe(replace_single_with_double_quotes)
                )

    return df_clean


