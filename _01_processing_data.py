
import glob
import os

import pandas as pd
from vnstock.api.listing import Listing
from vnstock.api.quote import Quote
def get_data():

   OUT = "data/"
   os.makedirs(OUT, exist_ok=True)
   syms = Listing(source="VCI").symbols_by_exchange()
   for s in syms["symbol"]:
      df = Quote(symbol=s, source="VCI").history(start="2000-01-01", end="2026-08-01")
      df.to_parquet(f"{OUT}/{s}.parquet")

   panel = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{OUT}/*.parquet")])
   panel.to_parquet(f"{OUT}/ohlcv_naive.parquet")

