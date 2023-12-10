"""Eksploracja danych >> Odkrywanie predykcyjne
   Wymagane pakiety:
   pip install pandas
   pip install seaborn"""

__author__ = "Tomasz Potempa"
__copyright__ = "Katedra Informatyki"
__version__ = "1.0.0"

from sklearn.preprocessing import StandardScaler
import seaborn as sns


def normalize(df):
    """Normalizacja danych z wykorzystaniem standaryzacji statystycznej"""

    scaler = StandardScaler()
    return scaler.fit_transform(df)


def visualize(df, title, regression=False, grouping=None):
    """Tworzenie wykresów.
       Parametr df to dane w formie ramki danych."""

    colors = ["blue", "orange", "red", "green", "magenta", "grey", "yellow", "black", "purple", "navy", "pink", "cyan", "white"]
    figure = sns.lmplot(x="liczba", y="wartosc", data=df, fit_reg=regression, legend=True, height=8.5, aspect=1.5, hue=grouping, palette=colors)
    figure.set_axis_labels("Liczba usług", "Wartość usług")
    figure.set(title=title)
