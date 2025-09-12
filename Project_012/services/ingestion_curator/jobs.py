import pandas as pd
def pii_scrub(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not any(k in c.lower() for k in ["ssn","password","secret"])]
    return df[cols]
def chunk_texts(texts, chunk_size=200):
    out = []; 
    for t in texts:
        for i in range(0, len(t), chunk_size): out.append(t[i:i+chunk_size])
    return out
