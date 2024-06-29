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

def weibull_dist_plot(k, c, n, ax=None, **kwargs):
    """
    Creates a distribution chart based on Weibull with k shape and c scale parameters to a n-length range of data.
    """
    # Style dictionary
    default_style = {"plot_color" : "blue", "ylabel" : "Frequency", "xlabel" : "Wind speed (m/s)", "title" : ""}
    for keywords in kwargs:
        if keywords in default_style.keys():
            default_style.update({keywords : kwargs[keywords]})

    # x, y values
    x = np.linspace(0, n, 1000)
    weibull = [((k / c) * ((n / c)**(k - 1)) * np.exp(-(n / c)**k)) for n in x]

    # Plot settings
    _, ax = plt.subplots(nrows=1, ncols=1, dpi=110)
    ax.plot(x, weibull, color=default_style["plot_color"])
    ax.set_title(default_style["title"])
    ax.set_ylabel(default_style["ylabel"])
    ax.set_xlabel(default_style["xlabel"])
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
        unconverted["Fecha"] = pd.to_datetime(unconverted["Fecha"])

        try:
            unconverted["Mes"] = unconverted["Fecha"].dt.month
            unconverted["Hora"] = unconverted["Fecha"].dt.hour
            if year:
                unconverted["Year"] = unconverted["Fecha"].dt.year

        except AttributeError:
            raise ValueError("Can only use .dt accessor with datetimelike values, your column is not datetimelike")

        return unconverted
    
    def hellman_column(self, height_zone, alpha_zone) -> pd.DataFrame:
        """
            Add
        """
        v_zone = self.pd_obj["Valor"]
        speed_values = f"Valor{height_zone}"
        self.pd_obj[speed_values] = hellman(
            v_0 = v_zone,
            h = height_zone,
            alpha = alpha_zone
        )
        return self.pd_obj
    
    def hourly_wind_plot(self, speed_values):
        """
            Add
        """

        plt.figure(figsize=(12, 6))
        sns.lineplot(x="Hora", y=speed_values, data=self.pd_obj)
        plt.title("Mean wind speed by hour")
        plt.xlabel("Hour")
        plt.xticks(range(0, 24))
        plt.grid(alpha=4/7)
        plt.show()
    
    def normal_wind_density_dist(self, speed_values, ax=None, **kwargs):
        """
            Plots the normal density distribution of wind speed.

            Parameters:
                speed_values (str): Column name containing wind speed data.
                **kwargs: Optional style parameters:
                    - ylabel (str): Label for the y-axis (default: "Frequency").
                    - xlabel (str): Label for the x-axis (default: "Wind speed (m/s)").
                    - title (str): Title of the plot (default: "Density distribution of wind speed").
                    - bins (int): Number of bins for the histogram (default: 20).
            
            Returns:
                None: The function plots the distribution on the provided or current axes.
        """
        
        # Default style dict.
        
        default_style = {"ylabel" : "Frequency", "xlabel" : "Wind speed (m/s)", "title" : "Density distribution of wind speed", "bins" : 20}
        default_style.update({key : kwargs[key] for key in kwargs if key in default_style})
        
        if ax is None:
            ax = plt.gca()
        
        sns.histplot(data=self.pd_obj[speed_values], bins=default_style["bins"], ax=ax)
        ax.set_title(default_style["title"])
        ax.set_ylabel(default_style["ylabel"])
        ax.set_xlabel(default_style["xlabel"])
    
    def weibull_plot(self, speed_values, ax=None, ran=25, **kwargs):
        """
            Plots the Weibull distribution for wind speed data.
        """
        
        k, c = weibull_parameters(v=self.pd_obj[speed_values])
        default_style = {"ylabel" : "Frequency", "xlabel" : "Wind speed (m/s)", "title" : ""}      
        default_style.update({key : kwargs[key] for key in kwargs if key in default_style})
        
        # x, y values
        x = np.linspace(0, ran, 1000)
        weibull = [((k / c) * ((n / c)**(k - 1)) * np.exp(-(n / c)**k)) for n in x]

        # Plot settings
        if ax is None:
            ax = plt.gca()
        
        ax.set_title(default_style["title"])
        sns.lineplot(x=x, y=weibull, ax=ax)  
        ax.set_ylabel(default_style["ylabel"])
        ax.set_xlabel(default_style["xlabel"])