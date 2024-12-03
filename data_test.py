data_test = {
    'Marque': ['Ineos'],
    'Modele': ['Grenadier'],
    'Annee': [2024],
    'Vehicule': ['Ineos Grenadier 3.0 T 286ch Fieldmaster Edition'],
    'Prix': [82490.0],
    'Date Publication': ['2022-04-01'],
    'Resumer': [
        '{"energie": "Essence", "puissance_commerciale": 286, "puissance_fiscale": 0, "consommation_mixte": 14.4, '
        '"emission_de_co2": 325, "boite_de_vitesses": "Automatique", "carrosserie": "4*4/SUV/Crossovers", '
        '"date_de_fin_de_commercialisation": "-", "Immatriculation": "7e546927-d3e3-477f-8971-b0cd70187264", '
        '"Object_Folder_Resumer": "Vehiculs/Models/INEOS/Resumer"}'
    ]
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)