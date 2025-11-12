from sentence_transformers import SentenceTransformer
import torch
import csv
import gzip
import dbm
import faiss
import pickle
import os
from time import perf_counter

DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"
DIMENSION = 768
BATCH_SIZE = 64

# ---------------------------
# Setup device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Remove index file if necessary
if os.path.isfile(INDEX_PATH):
    print(f"Removing {INDEX_PATH}")
    os.remove(INDEX_PATH)

start_time = perf_counter()

# ---------------------------
# Load the embedding model
# ---------------------------
print("Loading model...")
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

# ---------------------------
# Initialize FAISS GPU index
# ---------------------------
print("Initializing FAISS index on GPU...")
res = faiss.StandardGpuResources() if device == "cuda" else None
index_cpu = faiss.IndexFlatIP(DIMENSION)  # Inner product (for cosine similarity)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu) if device == "cuda" else index_cpu

# ---------------------------
# Initialize key-value store
# ---------------------------
print("Initializing key-value store...")
db = dbm.open(DB_PATH, "c")

# ---------------------------
# Sentence processing function
# ---------------------------
def process_sentences(sentences, attributes, start_index):
    start_processed = perf_counter()
    sentence_count = len(sentences)
    print(f"{start_index:,d}-{start_index + sentence_count - 1:,d}", end="", flush=True)

    embeddings = (
        model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
        )
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add vectors to FAISS GPU index
    index.add(embeddings)

    # Store quotes and metadata
    for i in range(sentence_count):
        key_bytes = str(start_index + i).encode("utf-8")
        value_bytes = pickle.dumps((sentences[i], attributes[i]))
        # Store them in the key-value store
        db[key_bytes] = value_bytes

    end_process = perf_counter()
    print(f":{end_process - start_processed:.1f}s ", end="", flush=True)

    # Periodic memory cleanup
    if (start_index + sentence_count) % (4 * BATCH_SIZE) == 0:
        torch.cuda.empty_cache()
        print("")

  # ---------------------------
# Read CSV and build index
# ---------------------------
print("Reading data, embedding, and storing...")

count = 0
with open("quotes.csv", "r") as file:
    quote_reader = csv.reader(file)
    next(quote_reader, None)  # Skip header

    sentences, attributes = [], []

    for row in quote_reader:
        sentences.append(row[0])
        attributes.append(row[1])

        if len(sentences) == BATCH_SIZE:
            process_sentences(sentences, attributes, count)
            count += len(sentences)
            sentences, attributes = [], []

    if len(sentences) > 0:
        process_sentences(sentences, attributes, count)
        count += len(sentences)

# ---------------------------
# Save results
# ---------------------------
print("\nWriting vector database and closing key-value store")
index_cpu_final = faiss.index_gpu_to_cpu(index)
# ---------------------------
# Write to Index and close database
# ---------------------------
faiss.write_index(index_cpu_final, INDEX_PATH)
db.close()

print(f"Wrote {count} quotes to {DB_PATH} and {INDEX_PATH}")
end_time = perf_counter()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")
