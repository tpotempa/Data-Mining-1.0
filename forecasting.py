"""Eksploracja danych >> Odkrywanie predykcyjne >> Prognozowanie
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install seaborn
   pip install plotly
   pip install prophet
   pip install statsmodels"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.5.0"

from data.mining.predictive import *
from data.datasources import *
from prophet import Prophet
from scipy import stats
from statsmodels.tools.eval_measures import rmse

import pandas as pd
import numpy as np
import datetime as dt
import warnings
import os

"""Zapytanie do analizy wolumenu tj. liczby sprzedaży wg dat"""
sales_volumen_by_date_query = "SELECT data, liczba " \
                              "FROM poczta_olap.sprzedaz_wg_dni_v";

"""Zapytanie do analizy wartości sprzedaży wg dat"""
sales_value_by_date_query = "SELECT data, wartosc " \
                            "FROM poczta_olap.sprzedaz_wg_dni_v";


def make_experiment_forecast(query):
    """Eksperyment prognozowania"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    # Utworzenie ramki danych
    # Wymagane jest aby ramka danych składała się z dwóch kolumn ds oraz y.
    # Kolumna ds tj. datestamp winna być datą w formacie YYYY-MM-DD albo YYYY-MM-DD HH:MM:SS.
    # Kolumna y winna być liczbą.
    df = pd.DataFrame(rs, columns=["ds", "y"])
    df["regressor"] = None
    print(df, os.linesep)

    x = "ds"
    y = "y"
    xlabel = "Dzień/Data"
    ylabel = "Wolumen/Wartość usług"
    title = "Sprzedaż usług"

    # Wizualizacja zbioru danych
    visualize_scatter(x=df[x], y=df[y], title=title, xlabel=xlabel, ylabel=ylabel)
    visualize_plot(x=df[x], y=df[y], title=title, xlabel=xlabel, ylabel=ylabel)

    # Eliminacja wartości odstających w oparciu o odchylenie standardowe
    sd_count = 3
    df_without_outliers = df[(np.abs(stats.zscore(df[y])) < sd_count)]

    # Wizualizacja zmodyfikowanego zbioru danych
    visualize_scatter(x=df_without_outliers[x], y=df_without_outliers[y], title=title, xlabel=xlabel, ylabel=ylabel)
    visualize_plot(x=df_without_outliers[x], y=df_without_outliers[y], title=title, xlabel=xlabel, ylabel=ylabel)

    # Utworzenie i wyświetlenie początkowych danych ZBIORU UCZĄCEGO
    df_train = df_without_outliers[df_without_outliers.ds <= dt.datetime.strptime("20191231", "%Y%m%d").date()]
    print("ZBIÓR UCZĄCY: ", df_train.head, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych ZBIORU TESTOWEGO
    df_test = df_without_outliers[df_without_outliers.ds > dt.datetime.strptime("20191231", "%Y%m%d").date()]
    print("ZBIÓR TESTOWY: ", df_test.head, os.linesep)

    print("Rozmiar ZBIORU UCZĄCEGO: ", df_train.loc[:, y].count() / df.loc[:, y].count())
    print("Rozmiar ZBIORU TESTOWEGO: ", df_test.loc[:, y].count() / df.loc[:, y].count(), os.linesep)

    # Utworzenie modelu prognozowania z wykorzystaniem algorytmu PROPHET
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="additive")

    # Dodanie regresora do modelu oraz wartości regresora do zbioru uczącego
    # df_train["niedziela"] = df_train["ds"].apply(sunday)
    # model.add_regressor("niedziela")

    # Uczenie modelu
    model.fit(df_train)

    # Utworzenie ramki danych dla okresu przyszłego tj. 1 roku
    periods = 365
    future = model.make_future_dataframe(periods=periods)

    # Dodanie wartości regresora do okresu przyszłego
    # future["niedziela"] = future["ds"].apply(sunday)

    # Prognozowanie dla okresu przyszłego
    forecast = model.predict(future)

    # Kolumny ramki danych z prognozą
    print(forecast.columns.values.tolist(), os.linesep)

    print("Rezultat prognozowania:")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(), os.linesep)

    # Ocena prognozowania
    predictions = forecast.iloc[-len(df_test):]["yhat"]
    actuals = df_test["y"]
    rms_error = round(rmse(predictions, actuals), 4)
    print("RMSE:", rms_error)

    # Wizualizacja prognozy i komponentów modelu
    visualize_prophet(model, forecast, xlabel, ylabel)

    # Walidacja krzyżowa
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="additive")
    model.fit(df_without_outliers)
    df_cv, df_metrics = model_cross_validation(model, test_period=365, training_period=2, period=0.1)
    print(df_cv.head)
    print(df_metrics)


"""Uruchamianie eksperymentów"""

"""Analiza wg wolumenu sprzedaży"""
make_experiment_forecast(sales_volumen_by_date_query)

"""Analiza wg wartości sprzedaży"""
# make_experiment_forecast(sales_value_by_date_query)