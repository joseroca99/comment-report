# Importing required modules
from flask import Flask, request, jsonify
import cohere
import requests
from qdrant_client import QdrantClient
import pandas as pd
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from flask_cors import CORS


df_eng = pd.read_csv("results.csv", names=['title','comment','rating'])
df_eng['language'] = 'English'
df_fre = pd.read_csv("results_french.csv", names=['title','comment','rating'])
df_fre['language'] = 'French'
df = pd.concat([df_eng, df_fre], axis=0)
df=df.dropna(subset=['comment']).reset_index(drop=True)

# Creating a flask app
app = Flask(__name__)
CORS(app)
# Initializing cohere client with API key
co = cohere.Client("tcVXdIMkW51aFc3Lg1JA8Iv0LRr5NqNzbq8HAtRO")


# Defining a route for the API call 
@app.route("/embed_and_search", methods=["POST"])
def embed_and_search():
    # Getting the text string from the request body
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Embedding the text string using cohere api
    embedding = co.embed(texts=[text], model="multilingual-22-12").embeddings

    vector_search = co.embed(texts=[text], model="multilingual-22-12").embeddings[0]

    # Performing vector search with qdrant
    qdrant_client = QdrantClient(
            url="https://f8748584-23d6-497d-8831-1265fbb636ff.us-east-1-0.aws.cloud.qdrant.io:6333", 
            api_key="z2qMGbIri894NVms_bN_LZe9DZeI4ExYBnxyf78cMaGP1q_vTCQeSw",
            )


    search_result = qdrant_client.search(
    collection_name="animal",
    query_vector=vector_search, 
    limit=50,
        )
    
    languages = set(df.loc[:,'language'])
    output = []
    for word in languages:
        search_result = qdrant_client.search(
        collection_name="test_collection",
        query_vector=vector_search, 
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchValue(value=word)
                )
            ]
        ),
        limit=3
        )

        result_vectors = [result.id for result in search_result]
        lang_comments = []
        for id in result_vectors:
            comment = df.loc[id,'comment']
            lang_comments.append(comment)
        dict_out = {"language":word, "comments":lang_comments}
        output.append(dict_out)

    # Returning the results as JSON
    return jsonify(output)

# Running the app
if __name__ == "__main__":
    app.run(debug=True)
