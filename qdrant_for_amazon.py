# Import modules
import pandas as pd
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Create Cohere client
co = cohere.Client("In30uIWlPuaZKU3hIaE0vjka8u0Vvd0MyPmgItAD")

# Create Qdrant client
client = QdrantClient("localhost", port=6333)

# Load dataframe
df_eng = pd.read_csv("results_french.csv", names=['title','comment','rating'])
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

# operation_info = client.upsert(
#     collection_name="animal",
#     wait=True,
#     points=points,
# )

searcher = input('search term: ')
vector_search = co.embed(texts=[searcher,], model="multilingual-22-12").embeddings[0]

search_result = client.search(
    collection_name="animal",
    query_vector=vector_search, 
    limit=5,
)
result_vectors = [result.id for result in search_result]
for id in result_vectors:

    print(df.loc[id,'comment'])