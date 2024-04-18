"""Eksploracja danych >> Normalizacja danych
   Wymagane pakiety:
   pip install scikit-learn"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.0.0"

from sklearn.preprocessing import StandardScaler


def normalize(df):
    """Normalizacja danych z wykorzystaniem standaryzacji statystycznej"""

    scaler = StandardScaler()
    return scaler.fit_transform(df)