from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_operations import ensure_collection, QdrantConfig
import os

load_dotenv()

cfg = QdrantConfig(
    url=os.environ["QDRANT_CONSOLE_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

client = QdrantClient(url=cfg.url, api_key=cfg.api_key)

ensure_collection(client, cfg)

print("Collection ready!")
print(client.get_collections())
