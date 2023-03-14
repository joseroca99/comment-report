from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)
# client.recreate_collection(
#     collection_name="test_collection2",
#     vectors_config=VectorParams(size=768, distance=Distance.DOT),
# )

collection_info = client.get_collection(collection_name="animal")

from qdrant_client.http.models import CollectionStatus

assert collection_info.status == CollectionStatus.GREEN
assert collection_info.vectors_count == 0

texts=[
    'Ride Him, Cowboy!',
    'Right Out of the Gate',
    'Elvis Has Left The Building',
    'On the Same Page',
    'Dropping Like Flies',
    'Elvis ha dejado el edificio',
    'En la misma pagina',
]
response = co.embed(texts=texts, model='multilingual-22-12')  
embeddings = response.embeddings # All text embeddings 
print(embeddings[0][:5]) # Print embeddings for the first text

from qdrant_client.http.models import PointStruct

points = []
for i in range(len(embeddings)):
    points.append(
        PointStruct(id=i, vector=embeddings[i],payload = {'phrase':texts[i]})
    )

operation_info = client.upsert(
    collection_name="animal",
    wait=True,
    points=points,
)

from qdrant_client.http.models import UpdateStatus

assert operation_info.status == UpdateStatus.COMPLETED

search_result = client.search(
    collection_name="animal",
    query_vector=embeddings[5], 
    limit=3
)