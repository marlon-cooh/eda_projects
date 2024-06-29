# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma

"""
    Utils function kit to calculate Wind speed at different heights, Weibull distribution based on a given speed range:
"""
def hellman(v_0, h, alpha, h_0=10):
    """
    Calculates wind speed in terms of height and wind speed at sea level.
    """
    v_80 = (h / h_0) ** alpha * v_0
    return v_80

def weibull_parameters(v):
    """
    Calculates wind speed in terms of height and wind speed at sea level.
    """
    k = (np.std(v) / np.mean(v)) ** (-1.086)
    c = np.mean(v) / gamma(1 + 1 / k)
    return k, c

def weibull_dist_plot(k, c, n, **kwargs):
    """
    Creates a distribution chart based on Weibull with k shape and c scale parameters to a n-length range of data.
    """
    # Style dictionary
    default_style = {"plot_color" : "blue", "ylabel" : "Frequency", "xlabel" : "Wind speed (m/s)"}
    for keywords in kwargs:
        if keywords in any(default_style.keys()):
            default_style.update({keywords : kwargs[keywords]})

    # x, y values
    x = np.linspace(0, n, 1000)
    weibull = [((k / c) * ((n / c)**(k - 1)) * np.exp(-(n / c)**k)) for n in x]

    # Plot settings
    plt.plot(x, weibull, color=default_style["plot_color"])
    plt.ylabel(default_style["ylabel"])
    plt.xlabel(default_style["xlabel"])
    plt.tight_layout()
    plt.show()

"""
    Pandas wclean extension to clean IDEAM data
"""
try:
    del pd.DataFrame.wclean
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("wclean")
class WindCleanMethods:
    def __init__(self, pd_obj):
        self.pd_obj = pd_obj
    
    def useful_columns(self) -> pd.DataFrame:
        """
            Method `useful_columns` filters IDEAM most relevant columns to depict in a dataframe to get a straight view of studied dataframe.
            
            Returns: 
            DataFrame referencing "NombreEstacion", "Categoria", "Municipio", "DescripcionSerie", "Frecuencia", "Valor", "Fecha"
        """
        useful_df = self.pd_obj.select_columns(
                    "NombreEstacion",
                    "Categoria",
                    "Municipio",
                    # "FechaInstalacion", # Not useful for our analysis
                    # "FechaSuspension", # Not useful for our analysis
                    "DescripcionSerie",
                    "Frecuencia",
                    "Valor",
                    "Fecha"
                )
            
        focus_df = useful_df[useful_df["DescripcionSerie"] == "Velocidad del viento de las 24 horas"]
        return focus_df
    
    def time_convert(self, year=False) -> pd.DataFrame:
        """
            Method `time_convert` takes input date.datetime like values and retrieves new columns within original dataset, thus, enabling time analysis.
            
            Parameters:
            year(bool) is False by default: if True, includes the 'Year' column.
        """
        unconverted = self.pd_obj.wclean.useful_columns()

        try:
            unconverted["Mes"] = unconverted["Fecha"].dt.month
            unconverted["Hora"] = unconverted["Fecha"].dt.hour
            if year:
                unconverted["Year"] = unconverted["Fecha"].dt.year

        except AttributeError:
            raise ValueError("Can only use .dt accessor with datetimelike values, your column is not datetimelike")

        return unconverted
    
    def hellman_column(self, height_zone, v_zone, alpha_zone) -> pd.DataFrame:
        """
            Add
        """
        v_zone = self.pd_obj["Valor"]
        speed_values = f"Valor{height_zone}"
        hellman_column = self.pd_obj[speed_values] = hellman(
            v_0 = v_zone,
            height = height_zone,
            alpha = alpha_zone
        )
        return hellman_column
    
    def hourly_wind_plot(self, speed_values):
        """
            Add
        """

        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Hour", y=speed_values, data=self.pd_obj.interpolate(method="linear"))
        plt.title("Mean wind speed by hour")
        plt.xlabel("Hour")
        plt.xticks(range(0, 24))
        plt.grid(alpha=4/7)
        plt.show()
    
    def normal_wind_density_dist(self, speed_values):
        """
            Add
        """
        sns.kdeplot(self.pd_obj[speed_values])
        plt.title("Density distribution of wind speed")
        plt.xlabel("Wind speed")
        plt.show()
    
    def weibull_plot(self, speed_values, ran=25):
        """
            Add
        """
        
        shape, scale = weibull_parameters(v=self.pd_obj[speed_values])
        return weibull_dist_plot(k=shape, c=scale, n=ran)