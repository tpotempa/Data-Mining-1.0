"""Eksploracja danych >> Odkrywanie deskrypcyjne >>> Grupowanie"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.5.0"

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

analysis_purchase_by_sender_query = "SELECT imie || ' ' || nazwisko AS nadawca, " \
                                    "EXTRACT(YEAR FROM data_urodzenia)::integer AS rok,  SUM(cena)::integer AS wartosc " \
                                    "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                                    "NATURAL JOIN poczta_olap.usluga " \
                                    "GROUP BY 1, 2 " \
                                    "ORDER BY 1, 2"

analysis_purchase_by_sender_query = "SELECT imie || ' ' || nazwisko AS nadawca, " \
                                    "EXTRACT(YEAR FROM data_urodzenia)::integer AS rok,  " \
                                    "CASE plec " \
                                    "WHEN 'M' THEN 1 " \
                                    "WHEN 'K' THEN 2 " \
                                    "ELSE NULL " \
                                    "END AS plec, " \
                                    "COUNT(cena)::integer AS liczba, " \
                                    "SUM(cena)::integer AS wartosc " \
                                    "FROM poczta_olap.sprzedaz NATURAL JOIN poczta_olap.nadawca " \
                                    "NATURAL JOIN poczta_olap.usluga " \
                                    "GROUP BY 1, 2, 3" \
                                    "ORDER BY 1, 2, 3"


def make_experiment_central_clustering(algorithm, number_of_clusters, query, n_init=25):
    """Eksperyment grupowania algorytmami środkowymi
       Parametr algorithm określa algorytm grupowania {kmeans, kmedoids}
       Parametr number_of_clusters określa liczbę skupień/grup {2, 3, ..., n}
       Parametr n_init określa liczbę uruchomień algorytmu z różnymi początkowymi środkami skupień"""
    warnings.filterwarnings("ignore")

    rs = connect(query)

    #df = pd.DataFrame(data=rs, columns=["wojewodztwo", "liczba", "wartosc"])
    #df = df.set_index("wojewodztwo")
    #df = pd.DataFrame(data=rs, columns=["nadawca", "rok", "wartosc"])
    #df = df.set_index("nadawca")
    df = pd.DataFrame(data=rs, columns=["nadawca", "rok", "plec", "liczba", "wartosc"])
    df = df.set_index("nadawca")
    print(df, os.linesep)

    scaled = normalize(df)
    #df_scaled = pd.DataFrame(data={"wojewodztwo": df.index.values, "liczba": scaled[:, 0], "wartosc": scaled[:, 1]})
    #df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "rok": scaled[:, 0], "wartosc": scaled[:, 1]})
    df_scaled = pd.DataFrame(data={"nadawca": df.index.values, "rok": scaled[:, 0], "plec": scaled[:, 1], "liczba": scaled[:, 2], "wartosc": scaled[:, 3]})
    print(df_scaled, os.linesep)

    model, df_grouped, d = central_clustering(number_of_clusters, df_scaled, algorithm, n_init)
    print(d, os.linesep)
    print(df_grouped, os.linesep)

    for i in range(model.n_clusters):
        print("Grupa", i)
        print(df_grouped[df_grouped.grupa == i], os.linesep)

    visualize(model, df_grouped)
    determine_optimal_number_of_groups(model, scaled)

    evaluate(model, scaled)
    evaluate_on_silhouette(model, scaled)
    evaluate_on_intercluster_distance_maps(model, scaled)


def make_experiment_hierarchical_clustering(number_of_clusters, query, linkage_method, metric):
    """Eksperyment grupowania
       Parametr number_of_clusters określa liczbę skupień/grup {2, 3, ..., n}
       Parametr method określa miarę odelgłości między skupieniami {single, complete, average, ward}"""
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
#make_experiment_central_clustering("kmedoids", 3, analysis_sale_by_region_query)
make_experiment_central_clustering("kmeans", 5, analysis_purchase_by_sender_query)
# make_experiment_hierarchical_clustering(3, analysis_sale_by_region_query, "ward", metric="euclidean")