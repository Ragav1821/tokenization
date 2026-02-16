from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
def chunk_text(text,chuck_size=40,overlap=10):
    word=text.split()
    chunks=[]
    for i in range(0, len(word),chuck_size - overlap):
        chunk=" ".join(word[i:i+ chuck_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)
query = input()
query_embedding = model.encode([query])
scores = cosine_similarity(query_embedding, chunk_embeddings)
print(scores)

top_index = np.argmax(scores)
print("Most Relevant Chunk:\n")
print(chunks[top_index])
