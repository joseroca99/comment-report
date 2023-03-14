# Import modules
import pandas as pd
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from pathlib import Path

# Create Cohere client
co = cohere.Client("In30uIWlPuaZKU3hIaE0vjka8u0Vvd0MyPmgItAD")

# Create Qdrant client
BASE_DIR = Path(__file__).parent.resolve()
KEY_DIR = BASE_DIR / 'etc' / 'my_key.txt'
with open(KEY_DIR) as f:
    SECRET_KEY = f.read().strip()

qdrant_client = QdrantClient(
    url="https://f8748584-23d6-497d-8831-1265fbb636ff.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key=SECRET_KEY,
)

# Load dataframe
df_eng = pd.read_csv("results.csv", names=['title','comment','rating'])
df_eng['language'] = 'English'
df_fre = pd.read_csv("results_french.csv", names=['title','comment','rating'])
df_fre['language'] = 'French'
df = pd.concat([df_eng, df_fre], axis=0)
df=df.dropna(subset=['comment']).reset_index(drop=True)

commentList = list(df.loc[:,'comment'])
embeddings = []
begin = 0
end = 90
while end < len(commentList):
    toEmbed = commentList[begin:end]
    embedding = co.embed(texts=toEmbed, model="multilingual-22-12").embeddings
    embeddings = embeddings + embedding
    begin += 90
    end += 90
toEmbed = commentList[begin::]
embedding = co.embed(texts=toEmbed, model="multilingual-22-12").embeddings
embeddings = embeddings + embedding

df['vector'] = embeddings

points = []
for index, row in df_eng.iterrows():
    points.append(
        PointStruct(id=index, vector=df.loc[index,'vector'], payload = {'language':df.loc[index,'language']})
    )

# operation_info = qdrant_client.upsert(
#     collection_name="animal",
#     wait=True,
#     points=points,
# )

searcher = input('search term: ')
vector_search = co.embed(texts=[searcher,], model="multilingual-22-12").embeddings[0]

search_result = qdrant_client.search(
    collection_name="animal",
    query_vector=vector_search, 
    limit=5,
)
result_vectors = [result.id for result in search_result]
for id in result_vectors:

    print(df.loc[id,'comment'])