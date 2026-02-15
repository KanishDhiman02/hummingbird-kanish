import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        if 'bug' in self.df.columns:
            self.y = (self.df['bug'] > 0).astype(int)
            cols_to_drop = ['bug', 'name', 'version', 'name.1', 'Name']
            self.X = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns])
        else:
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]
        
        self.feature_names = self.X.columns.tolist()
