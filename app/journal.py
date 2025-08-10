import pandas as pd
from typing import List, Dict

class TradeJournal:
    def __init__(self, path: str = "trades.csv"):
        self.path = path
        self.rows: List[Dict] = []

    def log(self, **kwargs):
        self.rows.append(kwargs)

    def to_csv(self):
        if self.rows:
            df = pd.DataFrame(self.rows)
            df.to_csv(self.path, index=False)
            return df
        return pd.DataFrame()
