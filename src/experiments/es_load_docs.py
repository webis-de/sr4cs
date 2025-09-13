"""Load documents into Elasticsearch index from a Parquet file."""

from elasticsearch import Elasticsearch, helpers
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

INDEX = "pierre-sr-boolq"
PARQUET = "../../data/final/refs.parquet"
API_KEY_ENCODED = os.getenv("API_KEY_ENCODED")


es = Elasticsearch(
    "https://elasticsearch.srv.webis.de",
    api_key=API_KEY_ENCODED,
)

print(es.info())

df = pd.read_parquet(PARQUET, columns=["ref_id", "title_norm", "abstract"])
df = df.rename(columns={"title_norm": "title"})
df = df.where(pd.notnull(df), None)


def actions():
    for r in df.itertuples(index=False):
        yield {
            "_op_type": "index",
            "_index": INDEX,
            "_id": r.ref_id,
            "_source": {
                "ref_id": r.ref_id,
                "title": r.title or "",
                "abstract": r.abstract or "",
            },
        }


helpers.bulk(es, actions(), chunk_size=2000, request_timeout=120)
print("Indexed docs:", es.count(index=INDEX)["count"])
