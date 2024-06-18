import numpy as np
import pandas as pd
import re

class Drop:
    def __init__(self, pandas_obj) -> pd.DataFrame:
        self.pandas_obj = pandas_obj
        
    def drop_cols(cols, pandas_obj):
        return pandas_obj.drop(cols, axis=1, inplace=True)
        
    def convert_figures(cols, pandas_obj, reg=","):
        """
            convert_figures aids to replace values as: "1,234,567" to 1234567, in this case, final value is divided in $4127.50 (TRM COP->USD, 18th June 2024)
            arguments:
            `cols`: column value to be converted
            `reg` : regular expression to replace, by default = ","
            `pandas_obj` : original raw dataframe.
            
        """
        df = Drop.drop_cols(cols, pandas_obj)
        for col in cols:
            df[col] = np.array([( int(re.sub(reg, "", i)) / 4127.50 ) for i in df[col].values])

def run():
    pass

if __name__ == "__main__":
    run()