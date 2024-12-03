import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd


def plot_predictions_vs_actuals(
        results: pd.DataFrame,
        model_name: str,
        predicted_col: str = "Predicted",
        actual_col: str = "RealPrice",
        xlabel: str = 'Prix Réel',
        ylabel: str = 'Prix prédites',
        col_length: int = 6,
        row_lenght: int = 6,
) -> None:
    """
    Trace un graphique comparant les valeurs prédites aux valeurs réelles.

    :param col_length:
    :param row_lenght:
    :param results: DataFrame contenant les colonnes des valeurs prédites et réelles.
    :param predicted_col: Nom de la colonne contenant les valeurs prédites.
    :param actual_col: Nom de la colonne contenant les valeurs réelles.
    :param model_name: Titre du graphique.
    :param xlabel: Label de l'axe des x.
    :param ylabel: Label de l'axe des y.
    """
    results[predicted_col] = pd.to_numeric(results[predicted_col], errors='coerce')

    # Tracer le graphique
    plt.figure(figsize=(col_length, row_lenght))
    sns.regplot(data=results, x=actual_col, y=predicted_col, color='teal', marker='o')

    # Formatage des axes
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.title(f"Comparaison Prédites vs Réelles : {model_name}", fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
