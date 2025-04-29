import pandas 

class FeatureEngineering:
    def split_data(self,df:pd.DataFrame):
        x = df.drop(columns="case_status")
        y = df["case_status"]
        return x, y