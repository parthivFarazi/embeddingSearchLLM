from sentence_transformers import SentenceTransformer
import dbm
import faiss
import pickle
import torch
from time import perf_counter

DB_PATH = "quotes.db"
INDEX_PATH = "quotes.index"
INPUT_FILE = "input.txt"
BEST_K = 3  # Number of closest quotes to display

# ---------------------------
# Setup device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load model
# ---------------------------
print("Loading model...")
model = SentenceTransformer("google/embeddinggemma-300m", device=device)

# ---------------------------
# Load FAISS index
# ---------------------------
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

if device == "cuda":
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("FAISS index moved to GPU.")
    except Exception as e:
        print(f"Could not move FAISS index to GPU: {e}")
        print("Falling back to CPU FAISS.")
else:
    print("Running FAISS on CPU.")

# ---------------------------
# Open key-value store
# ---------------------------
print(f"Using key-value store of type {dbm.whichdb(DB_PATH)}")
db = dbm.open(DB_PATH, "r")

results_file = open("recent_found_indices.txt", "a")

# ---------------------------
# Read and process quotes from input.txt
# ---------------------------
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as input_file:
        for line_num, line in enumerate(input_file, 1):
            sample_quote = line.strip()
            
            # Skip empty lines
            if not sample_quote:
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing line {line_num}: '{sample_quote}'")
            print('='*60)
            
            start_time = perf_counter()

            # Encode the query sentence on the appropriate device
            embedding = (
                model.encode(
                    [sample_quote],
                    batch_size=1,
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
            faiss.normalize_L2(embedding)

            # Search top matches
            _, I = index.search(embedding, BEST_K)

            end_time = perf_counter()

            # Display results
            for i in range(BEST_K):
                idx = int(I[0][i])
                print(f"{idx}", file=results_file)

                sentence, attributes = pickle.loads(db[str(idx).encode("utf-8")])

                print(f"\n{idx}: '{sentence}'\n\t{attributes}")

            print(f"Time taken: {end_time - start_time:.3f} seconds")

except FileNotFoundError:
    print(f"Error: '{INPUT_FILE}' not found. Please create the file with your quotes.")
except Exception as e:
    print(f"Error reading input file: {e}")

results_file.close()
db.close()
print(f"\nProcessing complete. Results appended to 'recent_found_indices.txt'")
