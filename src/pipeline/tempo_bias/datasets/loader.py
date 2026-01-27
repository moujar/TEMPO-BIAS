import pandas as pd
from typing import Iterator, Dict, Any

class DatasetLoader:
    def __init__(self, path: str, required_columns: list = None):
        self.path = path
        self.required_columns = required_columns or ['sentence_id', 'template', 'entity']
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        if self.path.endswith('.csv'):
            df = pd.read_csv(self.path)
        elif self.path.endswith('.json'):
            df = pd.read_json(self.path)
        else:
            raise ValueError("Unsupported format. Use csv or json.")
            
        self._validate(df)
        return df
        
    def _validate(self, df: pd.DataFrame):
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")
            
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _, row in self.data.iterrows():
            yield row.to_dict()
            
    def __len__(self):
        return len(self.data)
