"""Eksploracja danych >> Odkrywanie predykcyjne >> Predykcja
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install seaborn
   pip install prophet"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.3.0"

from data.mining.predictive import *
from data.datasources import *
from prophet import Prophet
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

"""Zapytanie do analizy sprzedaży wg dni"""
analysis_sale_by_date_query = "SELECT make_date(rok::integer, miesiac::integer, dzien::integer) AS data, SUM(cena)::integer AS wartosc " \
                              "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                              "INNER JOIN poczta_olap.czas ON czas.id_czasu = sprzedaz.id_czasu_nadania " \
                              "WHERE rok BETWEEN 2017 AND 2019 " \
                              "GROUP BY 1 " \
                              "ORDER BY 1"


def make_experiment_prediction(query):
    """Eksperyment predykcji"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    # Utworzenie ramki danych
    df = pd.DataFrame(rs, columns=["rok", "miesiac", "liczba", "wartosc"])
    print(df, os.linesep)

    x_feature = "liczba"
    y_feature = "wartosc"
    xlabel = "Liczba usług"
    ylabel = "Wartość usług"
    title = "Sprzedaż usług"

# Wykres punktowy kompletnego zbioru pobranych danych
    visualize_plot_scatter(x_plot=None, y_plot=None, x_scatter=df.loc[:, x_feature], y_scatter=df.loc[:, y_feature],
                           title=title, xlabel=xlabel, ylabel=ylabel)

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR UCZĄCEGO
    df_train = df[df.rok <= 2016]
    print(df_train.head, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych ZBIÓR TESTOWEGO
    df_test = df[df.rok > 2016]
    print(df_test.head, os.linesep)

    print("Rozmiar ZBIORU UCZĄCEGO: ", df_train.loc[:, y_feature].count()/df.loc[:, y_feature].count())
    print("Rozmiar ZBIORU TESTOWEGO: ", df_test.loc[:, y_feature].count()/df.loc[:, y_feature].count(), os.linesep)

    # Wykres punktowy z oznaczeniem lat ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + "\n{Zbiór UCZĄCY}", grouping="rok")

    # Wykres punktowy z oznaczeniem lat ZBIÓR TESTOWY
    visualize(df=df_test, x=x_feature, y=y_feature, title=title + "\n{Zbiór TESTOWY}", grouping="rok")

    # Wykres punktowy ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + "\n{Zbiór UCZĄCY}", regression=False)

    # Wykres punktowy z regresją liniową ZBIÓR UCZĄCY
    visualize(df=df_train, x=x_feature, y=y_feature, title=title + " & " + "Regresja" + "\n{Zbiór UCZĄCY}",
              regression=True)

    # Współczynnik korelacji liniowej Pearsona
    pc = np.corrcoef(df_train[x_feature], df_train[y_feature])
    print("Współczynnik korelacji liniowej Pearsona:\n", pc, os.linesep)

    # Utworzenie modelu predykcji tj. predyktora z wykorzystaniem REGRESJI LINIOWEJ
    x_train = np.array(df_train[x_feature]).reshape((-1, 1))
    y_train = np.array(df_train[y_feature])
    model, description = predictor(x_train, y_train)
    print(description, os.linesep)

    # Wizualizacja modelu predykcji
    x = np.linspace(np.min(x_train), max(x_train), 5000)
    y = model.coef_ * x + model.intercept_
    chart_title = "MODEL PREDYKCJI\n" + title + " & " + "Regresja"
    visualize_plot_scatter(x_plot=x, y_plot=y, x_scatter=x_train, y_scatter=y_train,
                           title=chart_title, xlabel=xlabel, ylabel=ylabel)

    # Predykcja
    x_test = np.array(df_test[x_feature]).reshape((-1, 1))
    y_test = np.array(df_test[y_feature])
    df_prediction, description = prediction(model, x_test, y_test)
    print("Rezultat predykcji:")
    print(df_prediction, os.linesep)

    # Ocena [modelu] predykcji
    print(description, os.linesep)

    # Wizualizacja predykcji
    chart_title = "PREDYKCJA\n" + title + " & " + "Regresja"
    visualize_plot_scatter(x_plot=x_test, y_plot=df_prediction["y_pred"], x_scatter=x_test, y_scatter=y_test,
                           title=chart_title, xlabel=xlabel, ylabel=ylabel)

def make_experiment_forecast(query):
    """Eksperyment prognozowania"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    # Utworzenie ramki danych
    df = pd.DataFrame(rs, columns=["ds", "y"])
    print(df, os.linesep)

    x_feature = "ds"
    y_feature = "y"
    xlabel = "Data"
    ylabel = "Wartość usług"
    title = "Sprzedaż usług"

    # Wizualizacja danych
    plt.figure(figsize=(15, 10))
    plt.plot(df["ds"], df["y"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    # Utworzenie modelu prognozowania z wykorzystaniem algorytmu PROPHET
    model = Prophet()
    model.fit(df)

    # Utworzenie ramki danych dla okresu przyszłego tj. 1 roku
    future = model.make_future_dataframe(periods=365)
    future.tail()

    # Prognozowanie dla okresu przyszłego
    forecast = model.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

    # Wizualizacja prognozy
    figure = model.plot(forecast)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#    fig2 = m.plot_components(forecast)



"""Uruchamienie eksperymentów"""
#make_experiment_prediction(analysis_sale_by_year_query)
make_experiment_forecast(analysis_sale_by_date_query)
