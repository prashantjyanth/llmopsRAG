from fastapi import FastAPI, UploadFile
import pandas as pd
from jobs import pii_scrub, chunk_texts
app = FastAPI(title="Ingestion & Curation")
@app.post("/ingest")
async def ingest(file: UploadFile): 
    df = pd.read_csv(file.file); return {"rows": len(df), "cols": list(df.columns)}
@app.post("/curate")
async def curate(file: UploadFile):
    df = pd.read_csv(file.file); clean = pii_scrub(df)
    texts = clean.astype(str).agg(" ".join, axis=1).tolist(); chunks = chunk_texts(texts)
    return {"rows_in": len(df), "rows_out": len(chunks)}
