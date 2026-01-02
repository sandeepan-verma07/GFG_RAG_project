from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_CONSOLE_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

print(client.get_collections())
