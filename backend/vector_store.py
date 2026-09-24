import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_index_name = os.environ.get("PINECONE_INDEX_NAME", "research-index")
_index = _pc.Index(_index_name)


def _embed(text: str) -> list[float]:
    result = _pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "passage"}
    )
    return result.data[0].values


def upsert_texts(texts: list[str], metadata: list[dict], namespace: str = "research") -> int:
    vectors = []
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        unique_id = f"{namespace}-{i}-{abs(hash(text)) % 100000}"
        vectors.append({
            "id": unique_id,
            "values": _embed(text),
            "metadata": {"text": text, **meta}
        })
    _index.upsert(vectors=vectors, namespace=namespace)
    return len(vectors)


def search_texts(query: str, top_k: int = 5, namespace: str = "research") -> list[dict]:
    query_embedding = _pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query],
        parameters={"input_type": "query"}
    )
    results = _index.query(
        vector=query_embedding.data[0].values,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    hits = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        hits.append({
            "text": meta.get("text", ""),
            "score": match.get("score", 0),
            "source": meta.get("source", ""),
            "agent": meta.get("agent", ""),
        })
    return hits


def clear_namespace(namespace: str = "research") -> None:
    _index.delete(delete_all=True, namespace=namespace)
