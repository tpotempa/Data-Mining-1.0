"""Eksploracja danych >> Żródła danych
   Wymagane pakiety:
   pip install psycopg2-binary
   pip install python-decouple"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.1.0"

import psycopg2 as db
from decouple import config

"""Dane połączenia z bazą danych na serwerze Katedry Informatyki"""
HOST = config("DB_HOST")
PORT = config("DB_PORT")
DATABASE = config("DATABASE")
USER = config("DB_USER")
PASSWORD = config("DB_PASSWORD")


def connect(query, host=HOST, port=PORT, database=DATABASE, user=USER, password=PASSWORD):
    """Połączenie z bazą danych PostgreSQL"""

    con = db.connect(host=host, port=port, database=database, user=user, password=password)

    # Utworzenie kursora
    cursor = con.cursor()

    # Wykonanie zapytania
    cursor.execute(query)

    # Odczytanie wyników zapytania do kolekcji w formie listy krotek
    rs = cursor.fetchall()

    # Zamknięcie kursora oraz połączenia
    cursor.close()
    con.close()

    return rs