import pandas as pd


class TradeJournal:
    def __init__(self, path: str = "trades.csv"):
        self.path = path
        self.rows: list[dict] = []

    def log(self, **kwargs):
        self.rows.append(kwargs)

    def to_csv(self):
        if self.rows:
            df = pd.DataFrame(self.rows)
            df.to_csv(self.path, index=False)
            return df
        return pd.DataFrame()
