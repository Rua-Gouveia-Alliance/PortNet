import pandas as pd
import numpy as np
import requests, re
from io import BytesIO, StringIO
import pdfplumber

WIKI_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
FTSE_HISTORY_PDF = "https://www.lseg.com/content/dam/ftse-russell/en_us/documents/policy-documents/ftse-100-constituent-history.pdf"

end = pd.Timestamp.today().normalize()
start = end - pd.DateOffset(years=3)

headers = {"User-Agent": "Mozilla/5.0"}

html = requests.get(WIKI_URL, headers=headers); html.raise_for_status()
tables = pd.read_html(StringIO(html.text))

def is_constituents_table(df):
    cols = {str(c).strip().lower() for c in df.columns}
    return ("company" in cols) and (("ticker" in cols) or ("epic" in cols))

const = next(t for t in tables if is_constituents_table(t)).copy()
const.columns = [str(c).strip() for c in const.columns]
company_col = "Company"
ticker_col  = "Ticker" if "Ticker" in const.columns else "EPIC"

current = const[[company_col, ticker_col]].copy()
current[company_col] = current[company_col].astype(str).str.strip()
current[ticker_col]  = current[ticker_col].astype(str).str.strip()

# filter out added and remove stocks
resp = requests.get(FTSE_HISTORY_PDF, headers=headers); resp.raise_for_status()
pdf_bytes = BytesIO(resp.content)

# match rows that start with a date like 22-Sep-25 or 7-Jun-24
date_re = re.compile(r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{2})\s+(.*)$")

rows = []
with pdfplumber.open(pdf_bytes) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            m = date_re.match(line)
            if not m:
                continue
            date_str, rest = m.groups()
            # PDF has two columns (Added | Deleted)
            parts = re.split(r"\s{2,}", rest.strip())
            if len(parts) >= 2:
                added = parts[0].strip()
                deleted = parts[1].strip()
            else:
                # fallback
                m2 = re.search(r"\s{3,}", rest)
                if m2:
                    cut = m2.start()
                    added = rest[:cut].strip()
                    deleted = rest[m2.end():].strip()
                else:
                    # fallback2 try to split on the last two-capitals token boundary
                    tokens = rest.split()
                    mid = len(tokens)//2
                    added = " ".join(tokens[:mid]).strip()
                    deleted = " ".join(tokens[mid:]).strip()
            rows.append((date_str, added, deleted))

changes = pd.DataFrame(rows, columns=["Date","Added","Deleted"])

# clean + normalize
def clean_name(s: str) -> str:
    s = (s or "").strip()
    s = s.split("(")[0]          # drop trailing tickers/notes
    s = s.replace("–","-").strip(" -\u200b").strip()

    for suf in [" plc", " Plc", " PLC"]:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return " ".join(s.split())

changes["Date"] = pd.to_datetime(changes["Date"], errors="coerce", dayfirst=True)
changes["Added"]  = changes["Added"].map(clean_name)
changes["Deleted"] = changes["Deleted"].map(clean_name)

# filter window and build sets
mask = changes["Date"].between(start, end, inclusive="both")
adds    = set(changes.loc[mask, "Added"].dropna().replace("", np.nan).dropna().str.lower().unique())
deletes = set(changes.loc[mask, "Deleted"].dropna().replace("", np.nan).dropna().str.lower().unique())

current["__norm"] = current[company_col].map(lambda x: clean_name(x).lower())
survivor_mask = (~current["__norm"].isin(adds)) & (~current["__norm"].isin(deletes))
survivors = (
    current.loc[survivor_mask, [ticker_col]]
           .drop_duplicates()
           .sort_values(by=ticker_col)
           .rename(columns={ticker_col: "Symbol"})
)

out_path = "../data/datasets/ftse100_survivors.csv"
survivors.to_csv(out_path, index=False)

print("\nSaved CSV: ../data/datasets/ftse100_survivors.csv")
