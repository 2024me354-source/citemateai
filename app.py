import streamlit as st
import fitz  # PyMuPDF
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Session state init ────────────────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()  # track filenames already ingested

# ── Secrets ──────────────────────────────────────────────────────────────────
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
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

model     = load_model()
supabase  = get_supabase()
groq_client = get_groq()

# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, min_words: int = 500, max_words: int = 800) -> list[str]:
    """Split text into chunks of roughly min_words–max_words words."""
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if len(current) >= min_words or (not chunks and current):
        chunks.append(" ".join(current))
    elif current and chunks:
        chunks[-1] += " " + " ".join(current)
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Generate normalized 384-dim embeddings."""
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def already_ingested(filename: str) -> bool:
    """Check Supabase if any chunks for this filename exist."""
    result = supabase.table("documents").select("id").eq("source_file", filename).limit(1).execute()
    return len(result.data) > 0


def ingest_pdf(file_bytes: bytes, filename: str):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
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

    progress = st.progress(0, text="Embedding chunks…")
    total = len(all_chunks)

    for idx, chunk in enumerate(all_chunks):
        vec = embed([chunk["content"]])[0]
        supabase.table("documents").insert({
            "content":     chunk["content"],
            "embedding":   vec,
            "page_number": chunk["page_number"],
            "source_file": filename,
        }).execute()
        progress.progress((idx + 1) / total, text=f"Storing chunk {idx+1}/{total}…")

    progress.empty()
    return total


def retrieve(question: str, match_count: int = 5) -> list[dict]:
    vec = embed([question])[0]
    result = supabase.rpc("match_documents", {
        "query_embedding": vec,
        "match_count": match_count,
    }).execute()
    return result.data or []


def build_context(chunks: list[dict], word_limit: int = 1500) -> str:
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
    system_prompt = (
        "You are a research assistant. "
        "Answer ONLY using the provided context. "
        "Cite sources using the format (source_file, page_number)."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CiteMate AI", page_icon="📚", layout="centered")
st.title("📚 CiteMate AI – PDF Research Assistant")
st.caption("Upload research PDFs and ask questions. Answers come with citations.")

# ── Section 1: Upload ─────────────────────────────────────────────────────────
st.header("1 · Upload PDFs")

# Show already-ingested files from this session
if st.session_state.ingested_files:
    st.success(f"✅ Ingested this session: {', '.join(st.session_state.ingested_files)}")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    # Read all file bytes BEFORE any button click causes a rerun
    files_data = [(uf.name, uf.read()) for uf in uploaded_files]

    new_files = [f for f in files_data if f[0] not in st.session_state.ingested_files]

    if not new_files:
        st.info("All uploaded files have already been ingested in this session.")
    elif st.button(f"Ingest {len(new_files)} PDF(s)"):
        for filename, file_bytes in new_files:
            # Check Supabase for duplicates
            if already_ingested(filename):
                st.info(f"⏭ {filename} already exists in the database — skipping.")
                st.session_state.ingested_files.add(filename)
                continue
            with st.spinner(f"Processing **{filename}**…"):
                try:
                    n = ingest_pdf(file_bytes, filename)
                    st.session_state.ingested_files.add(filename)
                    st.success(f"✅ {filename} — {n} chunk(s) stored.")
                except Exception as e:
                    st.error(f"❌ {filename} failed: {e}")


# ── Section 2: Ask ────────────────────────────────────────────────────────────
st.header("2 · Ask a Question")
question = st.text_area("Your question", placeholder="What does the paper say about…?", height=100)

top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)

if st.button("Get Answer", disabled=not question.strip()):
    with st.spinner("Searching knowledge base…"):
        try:
            chunks = retrieve(question, match_count=top_k)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.stop()

    if not chunks:
        st.warning("No relevant content found. Upload some PDFs first.")
        st.stop()

    context = build_context(chunks)

    with st.spinner("Asking Groq…"):
        try:
            answer = ask_groq(context, question)
        except Exception as e:
            st.error(f"LLM error: {e}")
            st.stop()

    # ── Section 3: Answer ─────────────────────────────────────────────────────
    st.header("3 · Answer")
    st.markdown(answer)

    # ── Section 4: Citations ──────────────────────────────────────────────────
    st.header("4 · Source Chunks")
    for i, c in enumerate(chunks, start=1):
        sim_pct = round(c.get("similarity", 0) * 100, 1)
        with st.expander(f"[{i}] {c['source_file']} — page {c['page_number']}  (similarity {sim_pct}%)"):
            st.write(c["content"])

# ── Debug / Diagnostics ───────────────────────────────────────────────────────
st.divider()
with st.expander("🛠 Debug: Inspect Supabase documents table"):
    if st.button("Fetch all rows from Supabase"):
        try:
            data = supabase.table("documents").select("id, source_file, page_number, content").execute()
            rows = data.data or []
            if not rows:
                st.warning("Table is empty — no documents have been ingested yet.")
            else:
                st.success(f"Found {len(rows)} chunk(s) across all files.")
                # Summary by file
                from collections import Counter
                file_counts = Counter(r["source_file"] for r in rows)
                st.subheader("Chunks per file")
                for fname, count in file_counts.items():
                    st.write(f"• **{fname}** — {count} chunk(s)")
                # Full table preview
                st.subheader("All rows (content truncated)")
                st.dataframe(
                    [
                        {
                            "id": r["id"],
                            "source_file": r["source_file"],
                            "page_number": r["page_number"],
                            "content_preview": r["content"][:120] + "…" if len(r["content"]) > 120 else r["content"],
                        }
                        for r in rows
                    ]
                )
        except Exception as e:
            st.error(f"Supabase error: {e}")

    st.divider()
    if st.button("🗑 Delete ALL rows (reset database)", type="secondary"):
        try:
            supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            st.session_state.ingested_files = set()
            st.success("All rows deleted and session state cleared.")
        except Exception as e:
            st.error(f"Delete failed: {e}")
