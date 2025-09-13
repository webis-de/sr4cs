"""Export all documents from an Elasticsearch index to a bulk NDJSON file."""

from elasticsearch import Elasticsearch, helpers
import json
from dotenv import load_dotenv
import os

load_dotenv()

ES_URL = "https://elasticsearch.srv.webis.de"
INDEX = "pierre-sr-boolq"
API_KEY = os.getenv("API_KEY_ENCODED")
OUTFILE = "../../data/final/exp/es/pierre-sr-boolq.ndjson"

es = Elasticsearch(ES_URL, api_key=API_KEY, request_timeout=120)

scan = helpers.scan(
    es, index=INDEX, query={"query": {"match_all": {}}}, size=1000, scroll="2m"
)

with open(OUTFILE, "w", encoding="utf-8") as f:
    for hit in scan:
        doc = {"index": {"_index": INDEX, "_id": hit["_id"]}}
        f.write(json.dumps(doc) + "\n")
        f.write(json.dumps(hit["_source"], ensure_ascii=False) + "\n")

print(f"Wrote bulk file: {OUTFILE}")
