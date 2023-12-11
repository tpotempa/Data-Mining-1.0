"""Eksploracja danych >> Odkrywanie predykcyjne >> Predykcja
   Wymagane pakiety:
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install seaborn"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.0.0"

from data.mining.predictive import *
from data.datasources import *
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

import seaborn as sns
import sklearn.metrics
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


"""Zapytanie do analizy sprzedaży wg lat"""
analysis_sale_by_year_query = "SELECT rok::integer, miesiac::integer, COUNT(id_uslugi)::integer AS liczba, SUM(cena)::integer AS wartosc " \
                              "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                              "INNER JOIN poczta_olap.czas ON czas.id_czasu = sprzedaz.id_czasu_nadania " \
                              "WHERE rok BETWEEN 2008 AND 2021 " \
                              "GROUP BY rok, miesiac " \
                              "ORDER BY rok, miesiac"

def make_experiment_prediction(query):
    """Eksperyment predykcji
       Parametr n_init określa liczbę uruchomień algorytmu z różnymi początkowymi środkami skupień"""
    warnings.filterwarnings('ignore')

    rs = connect(query)

    # Utworzenie ramki danych
    df = pd.DataFrame(rs, columns=["rok", "miesiac", "liczba", "wartosc"])
    #df = df.set_index("wojewodztwo")
    print(df, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych zbioru uczącego
    df_train = df[df.rok <= 2016]
    print(df_train.head, os.linesep)

    # Utworzenie i wyświetlenie początkowych danych zbioru testowego
    df_test = df[df.rok > 2016]
    print(df_test.head, os.linesep)


    # Wykres punktowy
    plt.figure(figsize=(15,10))
    plt.plot("liczba", "wartosc", data = df_train, linestyle = "none", marker = "o")
    plt.xlabel("Liczba usług", fontsize = "12", horizontalalignment = "center")
    plt.ylabel("Wartość usług", fontsize = "12", horizontalalignment = "center")
    plt.title("Sprzedaż usług 2008-2016")
    plt.show()

    # Wykres punktowy z regresją liniową
    visualize(df=df_train, title="Sprzedaż usług 2008-2016 & Regresja {Zbiór UCZĄCY}", regression=True)

    # Wykres punktowy z oznaczeniem lat ZBIÓR UCZĄCY
    visualize(df=df_train, title="Sprzedaż usług 2008-2016 {Zbiór UCZĄCY}", grouping="rok")

    # Wykres punktowy z oznaczeniem lat ZBIÓR TESTOWY
    visualize(df=df_test, title="Sprzedaż usług 2017-2021 {Zbiór TESTOWY}", grouping="rok")

    # Współczynnik korelacji liniowej Pearsona
    pc = np.corrcoef(df_train["liczba"], df_train["wartosc"])
    print("Współczynnik korelacji liniowej Pearsona = ", pc)


    # Uczenie
    x_train = np.array(df_train["liczba"]).reshape((-1, 1))
    y_train = np.array(df_train["wartosc"])
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)

    r_square = model.score(x_train, y_train)
    print("Parametry modelu:")
    print("- Współczynnik determinacji (r^2):", r_square)
    print("- Współczynnik a (slope):",  model.coef_)
    print("- Współczynnik b (intercept):",  model.intercept_)

    # Model
    plt.figure(figsize=(15, 10))
    x = np.linspace(200, 1300, 50000)
    y = model.coef_*x + model.intercept_
    plt.plot(x, y, color="blue", linewidth=1)

    plt.scatter(x_train, y_train, color="green")
    plt.show()



    # Predykcja
    x_test = np.array(df_test["liczba"]).reshape((-1,1))
    y_test = np.array(df_test["wartosc"])
    y_pred = model.predict(x_test)

   #mse = mean_squared_error(y_test, y_pred)

    print(y_test)
    print(y_pred)
    print("Ocena predykcji")
    #print("- Błąd średniokwadratowy (MSE):", mse)
    #print("- Współczynnik determinacji (r^2):", r2_score(y_test, y_pred), "|", model.score(x_test, y_test))


    # Wykres predykcji.
    plt.figure(figsize=(15, 10))
    plt.scatter(x_test, y_test, color = "green")
    plt.plot(x_test, y_pred, color = "blue", linewidth = 1)

    # Model
    #x = np.linspace(200,1300,50000)
    #y = model.coef_*x + model.intercept_
    #plt.plot(x, y, color = "red", linewidth = 1)

    plt.show()

"""Uruchamienie eksperymentów"""
make_experiment_prediction(analysis_sale_by_year_query)
