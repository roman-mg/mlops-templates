import os

import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", os.environ["OPENSEARCH_PASSWORD"]),
    use_ssl=True,
    verify_certs=False
)

index_name = "my_embeddings"
dimension = 768


def create_index() -> None:
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss"
                    }
                }
            }
        }
    }

    result = client.indices.create(index=index_name, body=index_body, ignore=400)
    print(result)


def add_embedding() -> None:
    embedding = np.random.rand(dimension).tolist()

    doc = {
        "text": "document example",
        "vector": embedding
    }

    resp = client.index(index=index_name, body=doc)
    print(resp)


def vector_search() -> None:
    query_vector = np.random.rand(dimension).tolist()

    search_body = {
        "size": 3,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector,
                    "k": 3
                }
            }
        }
    }

    results = client.search(index=index_name, body=search_body)
    for hit in results["hits"]["hits"]:
        print(hit["_source"]["text"], hit["_score"])


if __name__ == "__main__":
    create_index()
    add_embedding()
    vector_search()
