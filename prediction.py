"""Eksploracja danych >> Odkrywanie predykcyjne >> Predykcja
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install seaborn"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.2.0"

from data.mining.predictive import *
from data.datasources import *
import pandas as pd
import numpy as np
import warnings
import os


"""Zapytanie do analizy sprzedaży wg lat"""
analysis_sale_by_year_query = "SELECT rok::integer, miesiac::integer, COUNT(id_uslugi)::integer AS liczba, SUM(cena)::integer AS wartosc " \
                              "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                              "INNER JOIN poczta_olap.czas ON czas.id_czasu = sprzedaz.id_czasu_nadania " \
                              "WHERE rok BETWEEN 2008 AND 2019 " \
                              "GROUP BY rok, miesiac " \
                              "ORDER BY rok, miesiac"


def make_experiment_prediction(query):
    """Eksperyment predykcji"""
    warnings.filterwarnings('ignore')

    xlabel = "Liczba usług"
    ylabel = "Wartość usług"
    title = "Sprzedaż usług"

    rs = connect(query)

    # Utworzenie ramki danych
    df = pd.DataFrame(rs, columns=["rok", "miesiac", "liczba", "wartosc"])
    print(df, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR UCZĄCEGO
    df_train = df[df.rok <= 2016]
    print(df_train.head, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR TESTOWEGO
    df_test = df[df.rok > 2016]
    print(df_test.head, os.linesep)

    # Wykres punktowy wszystkich danych
    visualize_plot_scatter(x_plot=None, y_plot=None, x_scatter=df.iloc[:, 2], y_scatter=df.iloc[:, 3],
                           title=title, xlabel=xlabel, ylabel=ylabel)

    # Wykres punktowy z oznaczeniem lat ZBIÓR UCZĄCY
    visualize(df=df_train, x="liczba", y="wartosc", title="Sprzedaż usług 2008-2016 {Zbiór UCZĄCY}", grouping="rok")

    # Wykres punktowy z oznaczeniem lat ZBIÓR TESTOWY
    visualize(df=df_test, x="liczba", y="wartosc", title="Sprzedaż usług 2017-2019 {Zbiór TESTOWY}", grouping="rok")

    # Wykres punktowy ZBIÓR UCZĄCY
    visualize(df=df_train, x="liczba", y="wartosc", title="Sprzedaż usług 2008-2016 {Zbiór UCZĄCY}", regression=False)

    # Wykres punktowy z regresją liniową ZBIÓR UCZĄCY
    visualize(df=df_train, x="liczba", y="wartosc", title="Sprzedaż usług 2008-2016 & Regresja {Zbiór UCZĄCY}",
              regression=True)

    # Współczynnik korelacji liniowej Pearsona
    pc = np.corrcoef(df_train["liczba"], df_train["wartosc"])
    print("Współczynnik korelacji liniowej Pearsona:\n", pc, os.linesep)

    # Utworzenie modelu predykcji tj. predyktora z wykorzystaniem REGRESJI LINIOWEJ
    x_train = np.array(df_train["liczba"]).reshape((-1, 1))
    y_train = np.array(df_train["wartosc"])
    model, description = predictor(x_train, y_train)
    print(description, os.linesep)

    # Wizualizacja modelu predykcji
    x = np.linspace(400, 1200, 50000)
    y = model.coef_ * x + model.intercept_
    title = "MODEL PREDYKCJI\nSprzedaż usług 2008-2016 & Regresja"
    visualize_plot_scatter(x_plot=x, y_plot=y, x_scatter=x_train, y_scatter=y_train, title=title,
                           xlabel=xlabel, ylabel=ylabel)

    # Predykcja
    x_test = np.array(df_test["liczba"]).reshape((-1, 1))
    y_test = np.array(df_test["wartosc"])
    df_prediction, description = prediction(model, x_test, y_test)
    print("Rezultat predykcji:")
    print(df_prediction, os.linesep)

    # Ocena [modelu] predykcji
    print(description, os.linesep)

    # Wizualizacja predykcji
    title = "PREDYKCJA\nSprzedaż usług 2017-2019 & Regresja"
    visualize_plot_scatter(x_plot=x_test, y_plot=df_prediction["y_pred"], x_scatter=x_test, y_scatter=y_test,
                           title=title, xlabel=xlabel, ylabel=ylabel)


"""Uruchamienie eksperymentów"""
make_experiment_prediction(analysis_sale_by_year_query)
