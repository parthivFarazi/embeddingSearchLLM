
# 🧠 Embedding-Based Semantic Quote Search (PACE Supercomputing Project)

## 📘 Overview
This project implements a **semantic vector search system** over a large dataset of quotes (~493,000 total), using **Hugging Face sentence embeddings** and a **FAISS** vector index for efficient similarity retrieval.  

It was executed on **Georgia Tech’s PACE supercomputing cluster**, leveraging GPU acceleration and batch job scheduling with **SLURM**.  

The workflow demonstrates how large-scale text embeddings can be computed, indexed, and queried on distributed compute systems — replicating the foundation of modern search and recommendation engines.

---

## ⚙️ System Architecture

### 1️⃣ `make_index.py`
Builds the full FAISS index of embeddings.

- Loads the quotes dataset (`quotes.csv`)
- Uses the Hugging Face embedding model `google/embeddinggemma-300m`
- Embeds all quotes in GPU batches
- Stores results in:
  - `quotes.index` → FAISS vector index
  - `quotes.db` → SQLite database mapping IDs to quote text and authors

### 2️⃣ `find_quote.py`
Searches the index for the closest quotes to a given query.

- Loads `quotes.index` and `quotes.db`
- Reads query sentences from `input.txt`
- Encodes each query into an embedding
- Performs top-3 FAISS nearest-neighbor search
- Writes retrieved indices to `recent_found_indices.txt`

---

## 💻 HPC Setup (PACE Cluster)

### Job submission
Two SLURM batch scripts manage GPU execution:
- `job_gpu_make_index.sh` → builds embeddings on GPU  
- `job_gpu_find_quote.sh` → queries the index using pre-built embeddings  

Each job requests one GPU and up to 64 GB of memory:

```bash
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=2:00:00
Authentication
Hugging Face authentication is handled securely via a stored token:
export HF_TOKEN=$(cat ~/.hf_token)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
Memory optimization
To prevent CUDA fragmentation:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
---

## 🧩 Models Used
Model	Purpose	Notes
google/embeddinggemma-300m	Text embedding model	Compact and GPU-efficient
sentence-transformers	Embedding pipeline framework	Provides batching & normalization
faiss-gpu	Vector database	Enables sub-millisecond similarity search

---

## 📊 Dataset
Source: CSV of ~493 k English quotes and their authors.
Each row was embedded into a 768-dimensional vector and indexed via FAISS.


🧪 Example Queries
Input (input.txt):

```bash
You can't teach an old dog new tricks.
Practice makes perfect.
Look before you leap.
Birds of a feather flock together.
```
Output (Report.out):

```bash
Processing line 4: 'Birds of a feather flock together.'
407121: 'Birds of a feather flock together.' — English proverb
397140: 'Birds of a feather will flock together.' — Minsheu
52707:  'Find your flock and fly.' — Jennifer Coletta
```

Final indices (recent_found_indices.txt):
```bash
414986
408814
481488
397140
```
---

## 🧰 Tech Stack
- Language: Python 3.10
- Libraries: sentence-transformers, faiss-gpu, huggingface_hub, sqlite3
- Hardware: NVIDIA H100 GPU (PACE Cluster)
- Tools: SLURM, Conda/venv, PACE OnDemand

---

## 🚀 Results
- ✅ Embedded and indexed ~493 k quotes in ~25 minutes
- ✅ Queried multiple quotes with < 0.1 s latency per query
- ✅ Learned GPU memory management, tokenized auth, and HPC job orchestration
  
---

## 🧾 Key Files
### File	Description
- `make_index.py`	→ Builds the FAISS index and SQLite database
- `find_quote.py`	→ Searches for top-k nearest quotes
- `job_gpu_make_index.sh`	→ SLURM job for index creation
- `job_gpu_find_quote.sh` → SLURM job for query evaluation
- `requirements.txt` → Required Python packages
- `recent_found_indices.txt` → Output of retrieved quote indices
- `input.txt` → Input quotes for semantic search

---

## 💡 Learnings
- Managing GPU memory on large transformer models
- Using Hugging Face Hub tokens securely in HPC environments
- Efficient batching and FAISS indexing for large datasets
- SLURM scripting for multi-stage GPU workflows

