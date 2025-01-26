from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import faiss
import numpy as np
from gensim.models import KeyedVectors
import fasttext

app = FastAPI()

# CORS Configuration to allow frontend and Express backend to interact
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /data/ directory on Render for storing large files
TAG_VECTOR_FILE = "/data/tag_vector.pt"
TAG_LIST_FILE = "/data/tag_list.npy"
FASTTEXT_PATH = "/data/wiki-news-300d-1M.vec"
REDDIT_MODEL_PATH = "/data/music_vocab_embeddings.bin"
FAISS_INDEX_FILE = "/data/music_embeddings.index"

# Load the models and embeddings from persistent disk storage
tag_vector = torch.load(TAG_VECTOR_FILE, weights_only=True)
tag_list = np.load(TAG_LIST_FILE, allow_pickle=True).tolist()
fasttext_vectors = KeyedVectors.load_word2vec_format(FASTTEXT_PATH, binary=False)
model = fasttext.load_model(REDDIT_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_FILE)

def get_combined_embedding(word):
    if word in tag_list:
        idx = tag_list.index(word)
        return tag_vector[idx]
    elif word in model:
        return torch.tensor(model[word])
    elif word in fasttext_vectors:
        return torch.tensor(fasttext_vectors[word])
    return None

@app.get("/recommend/")
async def recommend(query: str = Query(..., min_length=1)):
    tokens = [token.strip() for token in query.split()]
    query_embeddings = [get_combined_embedding(token) for token in tokens if get_combined_embedding(token) is not None]

    if not query_embeddings:
        return {"query": query, "results": []}

    query_vector = torch.stack(query_embeddings).mean(0, keepdim=True).numpy().astype('float32')

    _, indices = index.search(query_vector, 10)
    results = [{"entity": tag_list[i], "score": float(1/(1 + _[0][j]))} for j, i in enumerate(indices[0])]

    return {"query": query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
