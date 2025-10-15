from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def on_bar(self, df: pd.DataFrame) -> dict | None:
        """
        Return:
          {"side": "BUY"/"SELL", "stop_pts": float, "tp_pts": float, "meta": {...}}
        or None
        """
        ...