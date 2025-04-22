"""Eksploracja danych >> Odkrywanie deskrypcyjne >>> Analiza skupień"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.6.0"

from data.mining.descriptive import *
from data.datasources import *
from data.normalization import *
import pandas as pd
import warnings
import os

"""Zapytanie do analizy sprzedaży wg województw"""
analysis_sale_by_region_query = "SELECT region, COUNT(id_uslugi)::integer AS liczba, SUM(cena)::integer AS wartosc " \
                                "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                                "INNER JOIN poczta_olap.czas ON czas.id_czasu = sprzedaz.id_czasu_nadania " \
                                "WHERE rok = 2019 " \
                                "GROUP BY region " \
                                "ORDER BY region"

"""Zapytanie do analizy sprzedaży wg klientów"""
analysis_purchase_by_sender_query = "WITH nadawcy AS " \
                                    "(SELECT imie || ' ' || nazwisko AS nadawca, " \
                                    "CASE plec WHEN 'M' THEN 1 WHEN 'K' THEN 2 ELSE NULL END AS plec, " \
                                    "region, " \
                                    "CAST(EXTRACT(YEAR FROM age(data_urodzenia)) AS integer) AS wiek, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE szybkosc = 'priorytetowa') / COUNT(*) AS integer) AS priorytet, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE szybkosc = 'ekonomiczna') / COUNT(*) AS integer) AS ekonomiczna, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka listowa') / COUNT(*) AS integer) AS list, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paczkowa') / COUNT(*) AS integer) AS paczka, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paczkomatowa') / COUNT(*) AS integer) AS paczkomat, " \
                                    "CAST(100.0 * COUNT(*) FILTER (WHERE rodzaj = 'przesyłka paletowa') / COUNT(*) AS integer) AS paleta, " \
                                    "CAST(COUNT(*) AS integer) AS liczba, " \
                                    "CAST(SUM(cena) AS integer) AS wartosc " \
                                    "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                                    "NATURAL JOIN poczta_olap.usluga " \
                                    "GROUP BY 1, 2, 3, 4 " \
                                    "ORDER BY 1, 2, 3) " \
                                    "SELECT nadawca, wiek, wartosc " \
                                    "FROM nadawcy " \
                                    "WHERE region = 'małopolskie'";

def make_experiment_central_clustering(algorithm, number_of_clusters, query, n_init=1):
    """Eksperyment analizy skupień algorytmami środkowymi
       Parametr algorithm określa algorytm analizy skupień {kmeans, kmedoids}
       Parametr number_of_clusters określa liczbę skupień {2, 3, ..., n}
       Parametr n_init określa liczbę uruchomień algorytmu z różnymi początkowymi środkami skupień"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    # Zapytanie analysis_sale_by_region_query
    '''
    df = pd.DataFrame(data=rs, columns=["wojewodztwo", "liczba", "wartosc"])
    df = df.set_index("wojewodztwo")
    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"wojewodztwo": df.index.values, "liczba": scaled[:, 0], "wartosc": scaled[:, 1]})
    '''

    # Zapytanie analysis_purchase_by_sender_query - Wariant I
    df = pd.DataFrame(data=rs, columns=["nadawca", "wiek", "wartosc"])
    df = df.set_index("nadawca")
    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "wiek": scaled[:, 0], "wartosc": scaled[:, 1]})

    # Zapytanie analysis_purchase_by_sender_query - Wariant II
    '''
    df = pd.DataFrame(data=rs, columns=["nadawca", "rok", "priorytet", "wartosc"])
    df = df.set_index("nadawca")
    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "rok": scaled[:, 0], "priorytet": scaled[:, 1], "wartosc": scaled[:, 2]})
    '''

    print(df, os.linesep)
    print(df_scaled, os.linesep)

    model, df_grouped, d = central_clustering(number_of_clusters, df_scaled, algorithm, n_init)
    print(d, os.linesep)
    print(df_grouped, os.linesep)

    for i in range(model.n_clusters):
        print("Grupa", i)
        print(df_grouped[df_grouped.grupa == i], os.linesep)

    visualize(model, df_grouped)
    determine_optimal_number_of_groups(model, scaled, 5)

    # Ewaluacja modeli
    evaluate(model, scaled)
    evaluate_on_silhouette(model, scaled)
    #evaluate_on_intercluster_distance_maps(model, scaled)


def make_experiment_hierarchical_clustering(number_of_clusters, query, linkage_method, metric):
    """Eksperyment grupowania
       Parametr number_of_clusters określa liczbę skupień {2, 3, ..., n}
       Parametr method określa miarę odległości między skupieniami {single, complete, average, ward}"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    df = pd.DataFrame(data=rs, columns=["wojewodztwo", "liczba", "wartosc"])
    df = df.set_index("wojewodztwo")
    print(df, os.linesep)

    scaled = normalize(df)
    df_scaled = pd.DataFrame(data={"wojewodztwo": df.index.values, "liczba": scaled[:, 0], "wartosc": scaled[:, 1]})
    print(df_scaled, os.linesep)

    visualize_dendrogram(scaled=scaled, linkage_method=linkage_method, metric=metric)
    model, df_grouped, d = hierarchical_clustering(k=number_of_clusters, df=df_scaled, linkage_method=linkage_method,
                                                   metric=metric)
    print(d, os.linesep)
    print(df_grouped, os.linesep)

    for i in range(model.n_clusters):
        print("Grupa", i)
        print(df_grouped[df_grouped.grupa == i], os.linesep)

    visualize_dendrograms(scaled, metric)


"""Uruchamienie eksperymentów"""
make_experiment_central_clustering("kmedoids", 3, analysis_purchase_by_sender_query)
#make_experiment_central_clustering("kmeans", 3, analysis_purchase_by_sender_query)
#make_experiment_hierarchical_clustering(3, analysis_purchase_by_sender_query, "ward", metric="euclidean")