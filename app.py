import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from collections import Counter
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Session state ─────────────────────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# ── Secrets ───────────────────────────────────────────────────────────────────
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

model       = load_model()
supabase    = get_supabase()
groq_client = get_groq_client()

# ── Core helpers ──────────────────────────────────────────────────────────────

def vec_to_string(vec: list) -> str:
    """Python list → pgvector string literal '[0.1,0.2,...]'"""
    return "[" + ",".join(str(round(float(v), 8)) for v in vec) + "]"

def embed(texts: list) -> list:
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()

def chunk_text(text: str, max_words: int = 600) -> list:
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
        if " ".join(words[i : i + max_words]).strip()
    ]

def already_ingested(filename: str) -> bool:
    r = supabase.table("documents").select("id").eq("source_file", filename).limit(1).execute()
    return len(r.data) > 0

def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    doc   = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_number": i, "text": text})
    doc.close()

    all_chunks = []
    for p in pages:
        for chunk in chunk_text(p["text"]):
            all_chunks.append({"page_number": p["page_number"], "content": chunk})

    if not all_chunks:
        raise ValueError("No extractable text found in PDF.")

    total    = len(all_chunks)
    progress = st.progress(0, text="Embedding & storing...")

    for idx, chunk in enumerate(all_chunks):
        vec_str = vec_to_string(embed([chunk["content"]])[0])
        supabase.table("documents").insert({
            "content":     chunk["content"],
            "embedding":   vec_str,
            "page_number": chunk["page_number"],
            "source_file": filename,
        }).execute()
        progress.progress((idx + 1) / total, text=f"Chunk {idx+1}/{total}")

    progress.empty()
    return total

def retrieve_rpc(vec_str: str, match_count: int) -> list:
    result = supabase.rpc("match_documents", {
        "query_embedding": vec_str,
        "match_count":     match_count,
    }).execute()
    return result.data or []

def retrieve_fallback(vec: list, match_count: int) -> list:
    """Fetch all rows, rank by cosine similarity in Python."""
    rows = supabase.table("documents").select(
        "content, page_number, source_file, embedding"
    ).execute().data or []

    q = np.array(vec, dtype="float32")
    scored = []
    for row in rows:
        emb = row.get("embedding")
        if emb is None:
            continue
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",")]
        r   = np.array(emb, dtype="float32")
        sim = float(np.dot(q, r))
        scored.append({**row, "similarity": sim})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:match_count]

def retrieve(question: str, match_count: int = 5) -> tuple:
    """Returns (chunks, method_used)"""
    vec     = embed([question])[0]
    vec_str = vec_to_string(vec)

    try:
        chunks = retrieve_rpc(vec_str, match_count)
        if chunks:
            return chunks, "RPC"
    except Exception:
        pass

    chunks = retrieve_fallback(vec, match_count)
    return chunks, "Fallback (Python)"

def build_context(chunks: list, word_limit: int = 1500) -> str:
    parts, total = [], 0
    for c in chunks:
        words = c["content"].split()
        if total + len(words) > word_limit:
            remaining = word_limit - total
            if remaining > 0:
                parts.append(" ".join(words[:remaining]))
            break
        parts.append(c["content"])
        total += len(words)
    return "\n\n---\n\n".join(parts)

def ask_groq(context: str, question: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": (
                "You are a research assistant. "
                "Answer ONLY using the provided context. "
                "Cite sources as (source_file, page_number)."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CiteMate AI", page_icon="📚", layout="centered")
st.title("📚 CiteMate AI – PDF Research Assistant")
st.caption("Upload PDFs · Ask questions · Get cited answers")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_diag = st.tabs(["🏠 Main", "🛠 Diagnostics"])

# ══════════════════════════════════════════════════════════
# TAB 1 — MAIN
# ══════════════════════════════════════════════════════════
with tab_main:

    # ── Upload ────────────────────────────────────────────
    st.header("1 · Upload PDFs")

    if st.session_state.ingested_files:
        st.success(f"✅ Ingested this session: {', '.join(st.session_state.ingested_files)}")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        files_data = [(uf.name, uf.read()) for uf in uploaded_files]
        new_files  = [f for f in files_data if f[0] not in st.session_state.ingested_files]

        if not new_files:
            st.info("All uploaded files already ingested this session.")
        elif st.button(f"Ingest {len(new_files)} PDF(s)"):
            for filename, file_bytes in new_files:
                if already_ingested(filename):
                    st.info(f"⏭ {filename} already in DB — skipping.")
                    st.session_state.ingested_files.add(filename)
                    continue
                with st.spinner(f"Processing **{filename}**..."):
                    try:
                        n = ingest_pdf(file_bytes, filename)
                        st.session_state.ingested_files.add(filename)
                        st.success(f"✅ {filename} — {n} chunks stored.")
                    except Exception as e:
                        st.error(f"❌ {filename} failed: {e}")

    # ── Ask ───────────────────────────────────────────────
    st.header("2 · Ask a Question")
    question = st.text_area("Your question", placeholder="What does the paper say about...?", height=100)
    top_k    = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)

    if st.button("Get Answer", disabled=not question.strip()):
        with st.spinner("Searching..."):
            try:
                chunks, method = retrieve(question, match_count=top_k)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        if not chunks:
            st.error("❌ No content found in Supabase. Go to the **🛠 Diagnostics** tab to investigate.")
            st.stop()

        st.caption(f"Retrieved via: **{method}**")
        context = build_context(chunks)

        with st.spinner("Asking Groq..."):
            try:
                answer = ask_groq(context, question)
            except Exception as e:
                st.error(f"LLM error: {e}")
                st.stop()

        st.header("3 · Answer")
        st.markdown(answer)

        st.header("4 · Source Chunks")
        for i, c in enumerate(chunks, start=1):
            sim_pct = round(c.get("similarity", 0) * 100, 1)
            with st.expander(f"[{i}] {c['source_file']} — page {c['page_number']} ({sim_pct}% match)"):
                st.write(c["content"])


# ══════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════
with tab_diag:
    st.header("🛠 Step-by-Step Diagnostics")
    st.caption("Run each check in order to find exactly what's broken.")

    # ── Check 1: Row count ────────────────────────────────
    st.subheader("Check 1 — How many rows are in Supabase?")
    if st.button("Count rows"):
        try:
            rows = supabase.table("documents").select(
                "id, source_file, page_number, content"
            ).execute().data or []

            if not rows:
                st.error("❌ Table is EMPTY. The ingest step is failing silently.")
            else:
                st.success(f"✅ {len(rows)} chunk(s) found.")
                counts = Counter(r["source_file"] for r in rows)
                for fname, n in counts.items():
                    st.write(f"• **{fname}** — {n} chunks")
                st.dataframe([{
                    "file":    r["source_file"],
                    "page":    r["page_number"],
                    "preview": r["content"][:80] + "...",
                } for r in rows[:20]])
        except Exception as e:
            st.error(f"Supabase fetch error: {e}")

    st.divider()

    # ── Check 2: Embedding column ─────────────────────────
    st.subheader("Check 2 — Is the embedding column populated?")
    if st.button("Inspect embeddings"):
        try:
            rows = supabase.table("documents").select(
                "id, source_file, embedding"
            ).limit(3).execute().data or []

            if not rows:
                st.error("❌ No rows found at all.")
            else:
                for r in rows:
                    emb = r.get("embedding")
                    st.write(f"**{r['source_file']}** — embedding type: `{type(emb).__name__}`")
                    if emb is None:
                        st.error("  ❌ embedding is NULL — insert is missing the vector!")
                    elif isinstance(emb, str):
                        vals = emb.strip("[]").split(",")
                        st.success(f"  ✅ String format, {len(vals)} dims. First 5: {vals[:5]}")
                    elif isinstance(emb, list):
                        st.success(f"  ✅ List format, {len(emb)} dims. First 5: {emb[:5]}")
                    else:
                        st.warning(f"  ⚠️ Unexpected type: {type(emb)}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # ── Check 3: RPC call ─────────────────────────────────
    st.subheader("Check 3 — Does the RPC function work?")
    if st.button("Test match_documents RPC"):
        try:
            test_vec = vec_to_string(embed(["explain the paper"])[0])
            st.code(f"Sending vector (first 60 chars): {test_vec[:60]}...")
            result = supabase.rpc("match_documents", {
                "query_embedding": test_vec,
                "match_count":     3,
            }).execute()
            rows = result.data or []
            if rows:
                st.success(f"✅ RPC returned {len(rows)} row(s)!")
                for r in rows:
                    st.write(f"• **{r['source_file']}** p{r['page_number']} — sim {round(r['similarity'], 3)}")
            else:
                st.warning("⚠️ RPC returned 0 rows. This is the ivfflat index problem.")
                st.info(
                    "Fix: Go to Supabase SQL Editor and run:\n\n"
                    "```sql\nDROP INDEX IF EXISTS idx_embedding;\n\n"
                    "CREATE INDEX idx_embedding ON public.documents\n"
                    "USING hnsw (embedding vector_cosine_ops);\n```\n\n"
                    "HNSW works with any number of rows. ivfflat needs 100+ rows to work reliably."
                )
        except Exception as e:
            st.error(f"RPC error: {e}")
            st.info("This means the function `match_documents` either doesn't exist or has a signature mismatch. Re-run the SQL setup script.")

    st.divider()

    # ── Check 4: Fallback ranking ─────────────────────────
    st.subheader("Check 4 — Does Python fallback ranking work?")
    if st.button("Test fallback retrieval"):
        try:
            vec    = embed(["explain the paper"])[0]
            chunks = retrieve_fallback(vec, match_count=3)
            if chunks:
                st.success(f"✅ Fallback returned {len(chunks)} chunk(s).")
                for c in chunks:
                    st.write(f"• **{c['source_file']}** p{c['page_number']} — sim {round(c['similarity'], 3)}")
            else:
                st.error("❌ Fallback also returned nothing — table is empty or embeddings are NULL.")
        except Exception as e:
            st.error(f"Fallback error: {e}")

    st.divider()

    # ── Reset ─────────────────────────────────────────────
    st.subheader("Reset")
    if st.button("🗑 Delete ALL rows", type="secondary"):
        try:
            supabase.table("documents").delete().neq(
                "id", "00000000-0000-0000-0000-000000000000"
            ).execute()
            st.session_state.ingested_files = set()
            st.success("All rows deleted. Session state cleared.")
        except Exception as e:
            st.error(f"Delete failed: {e}")
