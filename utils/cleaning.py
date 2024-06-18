import numpy as np
import pandas as pd
import re

class Drop:
    def __init__(self, pandas_obj) -> pd.DataFrame:
        self.pandas_obj = pandas_obj
        
    def drop_cols(self, cols):
        return self.pandas_obj.drop(cols, axis=1, inplace=True)
        
    def convert_figures(self, cols, reg=","):
        """
            convert_figures aids to replace values as: "1,234,567" to 1234567, in this case, final value is divided in $4127.50 (TRM COP->USD, 18th June 2024)
            arguments:
            `cols`: column value to be converted,
            `reg` : regular expression to replace, by default = ","        
        """
        for col in cols:
            self.pandas_obj[col] = self.pandas_obj[col].apply(lambda x: int(re.sub(reg, "", str(x))) / 4127.50)

def run():
    pass

if __name__ == "__main__":
    run()