from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid


class QdrantConfig:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str = "all_user_docs", 
        vector_size: int = 384,                    
        distance: str = "Cosine",     
    ):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance



def get_qdrant_client(cfg: QdrantConfig) -> QdrantClient:
    return QdrantClient(url=cfg.url, api_key=cfg.api_key)


def ensure_collection(client: QdrantClient, cfg: QdrantConfig) -> None:
    """
    Ensure collection exists with given vector configuration.
    """
    try:
        client.get_collection(cfg.collection_name)
    except Exception:
        client.create_collection(
            collection_name=cfg.collection_name,
            vectors_config=models.VectorParams(
                size=cfg.vector_size,
                distance=models.Distance(cfg.distance),
            ),
            hnsw_config=models.HnswConfigDiff(        
                payload_m=16,
                m=0,    
            ),
        )

        # Create tenant payload index on "user_id"
        client.create_payload_index(
            collection_name=cfg.collection_name,
            field_name="user_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                is_tenant=True,   
            ),
        )

        
        client.create_payload_index(
            collection_name=cfg.collection_name,
            field_name="doc_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )



def upsert_chunks(
    client: QdrantClient,
    cfg: QdrantConfig,
    user_id: str,
    filename: str,
    vectors: List[List[float]],
    chunks: List[str],
) -> int:
    """
    Upserts chunks into Qdrant with payload partitioning
    Uses user_id + filename as identifiers
    """
    points = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        point_id = str(uuid.uuid4())  
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "user_id": user_id,
                    "doc_id": filename,  
                    "filename": filename,
                    "page": chunk["page"],
                    "chunk_index": i,
                    "text": chunk["text"],
                },
            )
        )
    client.upsert(collection_name=cfg.collection_name, points=points, wait=True)
    return len(points)



def list_user_docs(
    client: QdrantClient,
    cfg: QdrantConfig,
    user_id: str,
    limit: int = 1000,
) -> List[Tuple[str, str]]:
    """
    Returns (doc_id, filename) pairs for user's stored docs
    """
    scroll_res = client.scroll(
        collection_name=cfg.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=limit,
        with_payload=True,
    )
    points = scroll_res[0] if isinstance(scroll_res, tuple) else scroll_res
    seen: Dict[str, str] = {}
    for p in points:
        payload = p.payload or {}
        did = payload.get("doc_id")
        fname = payload.get("filename")
        if did and did not in seen:
            seen[did] = fname or did
    return [(did, seen[did]) for did in seen]



def delete_document(
    client: QdrantClient,
    cfg: QdrantConfig,
    user_id: str,
    filename: str,
) -> None:
    """
    Deletes all points for given user_id + filename (doc_id).
    """
    print("DELETING...")
    client.delete(
        collection_name=cfg.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=filename)),
                ]
            )
        ),
        wait=True,
    )
    print("Deleted")




def search(
    client: QdrantClient,
    cfg: QdrantConfig,
    user_id: str,
    query_vector: List[float],
    limit: int = 5,
    filename: Optional[str] = None,
) -> List[Dict]:
    """
    Search Qdrant with query_vector.
    Scope by user_id, optionally restrict to filename (doc_id).
    Returns list of payload dicts.
    """
    must = [models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
    if filename:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=filename)))

    results = client.query_points(
        collection_name=cfg.collection_name,
        query=query_vector,
        limit=limit,
        query_filter=models.Filter(must=must),
        with_payload=True,
    )

    # Normalize return type
    if isinstance(results, tuple):
        points, _ = results
    elif hasattr(results, "points"):
        points = results.points
    else:
        points = results

    seen = set()
    unique_results = []
    for p in points:
        if not getattr(p, "payload", None):
            continue
        key = (p.payload.get("doc_id"), p.payload.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        unique_results.append({
            "doc_id": p.payload.get("doc_id"),
            "text": p.payload.get("text"),
            "score": getattr(p, "score", None),
            "page": p.payload.get("page"),
            "chunk_index": p.payload.get("chunk_index"),
            "filename": p.payload.get("filename"),
        })

    return unique_results
