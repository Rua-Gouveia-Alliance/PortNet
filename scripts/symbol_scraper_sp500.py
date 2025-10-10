import yfinance as yf
import pandas as pd
import numpy as np
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

start = pd.Timestamp("2022-09-30")   # inclusive window start
end   = pd.Timestamp("2025-09-30")   # inclusive window end

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(WIKI_URL, headers=headers)
response.raise_for_status()

tables = pd.read_html(response.text)
constituents = tables[0].copy()
constituents.columns = [c.strip() for c in constituents.columns]
current = constituents["Symbol"].astype(str).str.strip()

changes = tables[1].copy()
changes["Effective Date"] = pd.to_datetime(changes["Effective Date"].stack(), format="%B %d, %Y", errors="coerce").unstack()
changes_filtered = changes.loc[changes["Effective Date"]["Effective Date"].between(start, end, inclusive="both")]
changes_added = changes_filtered["Added"]["Ticker"]
changes_removed = changes_filtered["Removed"]["Ticker"]

survivors = current[~current.isin(set(changes_added))][~current.isin(set(changes_removed))].drop_duplicates().sort_values().to_list()
survivors = [s.replace(".", "-") for s in survivors]

pd.DataFrame({"Symbol": survivors}).to_csv("../data/sp500_survivors.csv", index=False)
print("\nSaved CSV: ../data/sp500_survivors.csv")
