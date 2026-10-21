# Data Engine - Basic Data Indexing for RAG

Pipeline: **load documents -> split into uniform chunks (~200 tokens) -> embed every chunk -> in-memory vector store -> inspect for consistency**

## Setup
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # optional: add GOOGLE_API_KEY for Gemini embeddings
```
Without a key the app uses a built-in local embedding model, so it always runs.

## Run
```
streamlit run app.py
```
1. Choose "Sample Gen AI collection" or upload your own PDF(s).
2. Click **Build index**.
3. Read the metric cards and the consistency checks, open "View all chunks".
4. Type a question in the bottom box to test retrieval (top 3 chunks + similarity).

## Terminal check
```
python rag_index.py
```
Last line: `[OK] Indexing pipeline executed end-to-end`

## Files
- `rag_index.py` - load, chunk, embed, store, inspect, search
- `app.py` - Streamlit UI
- `.streamlit/config.toml` - cream / olive theme
