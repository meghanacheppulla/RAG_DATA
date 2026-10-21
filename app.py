import html

import pandas as pd
import streamlit as st

from rag_index import (CHUNK_TOKENS, create_index, load_pdf, load_sample_docs, search)

st.set_page_config(page_title="Data Engine", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(160deg, #f1ebe0 0%, #fbf8f2 55%, #f4eee3 100%);}
    .block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 6rem;}
    .app-title {font-size: 2.3rem; font-weight: 800; color: #2b2f26; margin: 0;}
    .app-sub {text-align: right; color: #6b6f63; font-size: 0.85rem; padding-top: 0.9rem;}
    .card {background: #fffdf7; border: 1px solid #e6dfd0; border-radius: 12px; padding: 14px 18px;}
    .card .lbl {font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; color: #7a7d70;}
    .card .val {font-size: 1.45rem; font-weight: 800; color: #3d4a37; margin-top: 4px; word-break: break-all;}
    .sec-title {font-weight: 700; color: #2b2f26; margin: 1.6rem 0 0 0;}
    .sec-sub {font-size: 0.78rem; color: #7a7d70; margin-bottom: 0.7rem;}
    .qbox {background: #fffdf7; border: 1px solid #e6dfd0; border-radius: 10px; padding: 12px 16px; margin-bottom: 12px;}
    .hit {background: #fffdf7; border: 1px solid #e6dfd0; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;}
    .hit .meta {font-size: 0.75rem; color: #7a7d70; margin-bottom: 6px;}
    .hit .score {float: right; font-weight: 700; color: #4a5a42;}
    .pill {display: inline-block; background: #e3e9dd; color: #3d4a37; border-radius: 999px;
           padding: 2px 12px; font-size: 0.78rem; font-weight: 600;}
    .ok {color: #1a7f37; font-weight: 700;} .bad {color: #c0392b; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def card(label: str, value: str) -> str:
    return (f'<div class="card"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val">{html.escape(str(value))}</div></div>')


# ------------------------------------------------------------------ header
left, right = st.columns([1.2, 1])
left.markdown('<div class="app-title">Data Engine</div>', unsafe_allow_html=True)
right.markdown(
    '<div class="app-sub">Load documents, split them into uniform chunks, embed every chunk '
    "and inspect the in-memory vector store.</div>",
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------- workspace
with st.expander("📁  DOCUMENT WORKSPACE", expanded=True):
    source = st.radio(
        "Knowledge source",
        ["Sample Gen AI collection", "Upload PDF(s)"],
        horizontal=True,
    )
    files = []
    if source == "Upload PDF(s)":
        files = st.file_uploader("Choose PDF knowledge source(s)", type=["pdf"],
                                 accept_multiple_files=True) or []
    build = st.button("Build index", type="primary")

if build:
    try:
        with st.spinner("Loading, chunking and embedding..."):
            if source == "Upload PDF(s)":
                if not files:
                    raise ValueError("Please upload at least one PDF first.")
                docs, names = [], []
                for f in files:
                    docs += load_pdf(f, f.name)
                    names.append(f.name)
                if not docs:
                    raise ValueError("No selectable text found in the PDF (is it a scanned image?).")
                source_name = names[0] if len(names) == 1 else f"{len(names)} PDF files"
            else:
                docs = load_sample_docs()
                source_name = "Sample Gen AI collection"
            result = create_index(docs)
            result["source_name"] = source_name
            st.session_state["index"] = result
            st.session_state.pop("last_query", None)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build the index: {e}")

idx = st.session_state.get("index")

# ------------------------------------------------------------ metric cards
if idx:
    rep = idx["report"]
    n_docs = len({(d.metadata["source"], d.metadata["page"]) for d in idx["docs"]})
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Source", idx["source_name"]), unsafe_allow_html=True)
    c2.markdown(card("Total pages / docs", n_docs), unsafe_allow_html=True)
    c3.markdown(card("Divided chunks", len(idx["chunks"])), unsafe_allow_html=True)
    c4.markdown(card("Chunk size", f"{CHUNK_TOKENS} tokens"), unsafe_allow_html=True)
    st.write("")
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(card("Embeddings", rep["n_vectors"]), unsafe_allow_html=True)
    d2.markdown(card("Dimensions", rep["dimension"]), unsafe_allow_html=True)
    d3.markdown(card("Vector store", "In-memory"), unsafe_allow_html=True)
    passed = sum(c["passed"] for c in rep["checks"])
    d4.markdown(card("Consistency", f"{passed}/{len(rep['checks'])} passed"), unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:10px"><span class="pill">{html.escape(idx["embedding_label"])}</span></div>',
                unsafe_allow_html=True)
    if idx["note"]:
        st.info(idx["note"])

    # ------------------------------------------------------------ inspection
    st.markdown('<div class="sec-title">Index inspection</div>'
                '<div class="sec-sub">Consistency checks run on the stored chunks and vectors.</div>',
                unsafe_allow_html=True)
    rows = "".join(
        f'<tr><td style="padding:4px 10px"><span class="{"ok" if c["passed"] else "bad"}">'
        f'{"✔" if c["passed"] else "✘"}</span></td>'
        f'<td style="padding:4px 10px">{html.escape(c["check"])}</td>'
        f'<td style="padding:4px 10px;color:#7a7d70">{html.escape(c["detail"])}</td></tr>'
        for c in rep["checks"]
    )
    st.markdown(f'<div class="card"><table style="width:100%;font-size:0.9rem">{rows}</table></div>',
                unsafe_allow_html=True)
    st.caption(f"Chunk tokens - min {rep['min_tokens']}, average {rep['avg_tokens']}, max {rep['max_tokens']}")

    with st.expander("View all chunks"):
        df = pd.DataFrame([
            {
                "Chunk": c.metadata["chunk_id"],
                "Source": c.metadata["source"],
                "Page": c.metadata["page"],
                "Tokens": c.metadata["tokens"],
                "Characters": len(c.page_content),
                "Preview": c.page_content[:110] + ("..." if len(c.page_content) > 110 else ""),
            }
            for c in idx["chunks"]
        ])
        st.dataframe(df, hide_index=True)

# ---------------------------------------------------------- ask / retrieval
st.markdown('<div class="sec-title">✦ Ask your document</div>'
            '<div class="sec-sub">Test retrieval: the closest chunks from the in-memory vector store.</div>',
            unsafe_allow_html=True)

query = st.chat_input("Execute query...", disabled=not idx)
if not idx:
    st.info("Build the index first, then ask a question to test retrieval.")
if query and idx:
    try:
        st.session_state["last_query"] = (query, search(idx["store"], query, k=3))
    except Exception as e:  # noqa: BLE001
        st.error(f"Search failed: {e}")

if idx and "last_query" in st.session_state:
    q, hits = st.session_state["last_query"]
    st.markdown(f'<div class="qbox">👤 &nbsp;{html.escape(q)}</div>', unsafe_allow_html=True)
    for rank, (doc, score) in enumerate(hits, start=1):
        st.markdown(
            f'<div class="hit"><div class="meta"><span class="score">similarity {score:.3f}</span>'
            f'#{rank} · {html.escape(str(doc.metadata["source"]))} · page {doc.metadata["page"]} · '
            f'chunk {doc.metadata["chunk_id"]} · {doc.metadata["tokens"]} tokens</div>'
            f'{html.escape(doc.page_content)}</div>',
            unsafe_allow_html=True,
        )
