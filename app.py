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
    vec     = embed([question])[0]
    vec_str = vec_to_string(vec)
    try:
        chunks = retrieve_rpc(vec_str, match_count)
        if chunks:
            return chunks, "RPC"
    except Exception:
        pass
    chunks = retrieve_fallback(vec, match_count)
    return chunks, "Fallback"

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
# THEME & STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CiteMate AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

/* ── Root variables ── */
:root {
    --red:        #e8132a;
    --red-dim:    #a50e1e;
    --red-glow:   rgba(232,19,42,0.25);
    --bg:         #0c0c0f;
    --bg2:        #111118;
    --bg3:        #18181f;
    --border:     rgba(232,19,42,0.22);
    --border-dim: rgba(255,255,255,0.06);
    --text:       #e8e8f0;
    --muted:      #7a7a90;
    --mono:       'Share Tech Mono', monospace;
    --display:    'Rajdhani', sans-serif;
    --body:       'Exo 2', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

/* ── Animated grid background ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(232,19,42,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,19,42,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
    animation: gridshift 20s linear infinite;
}
@keyframes gridshift {
    0%   { background-position: 0 0; }
    100% { background-position: 48px 48px; }
}

/* ── Diagonal accent bar ── */
.citemate-hero {
    position: relative;
    width: 100%;
    background: linear-gradient(135deg, #0e0e14 0%, #13080b 50%, #0e0e14 100%);
    padding: 52px 60px 44px;
    overflow: hidden;
    border-bottom: 1px solid var(--border);
}
.citemate-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(232,19,42,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.citemate-hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
    animation: scanline 3s ease-in-out infinite;
}
@keyframes scanline {
    0%,100% { opacity: 0.4; }
    50%      { opacity: 1; }
}

.hero-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 3px;
    color: var(--red);
    text-transform: uppercase;
    border: 1px solid var(--border);
    padding: 4px 14px;
    margin-bottom: 18px;
    background: rgba(232,19,42,0.07);
}
.hero-title {
    font-family: var(--display);
    font-size: clamp(36px, 5vw, 68px);
    font-weight: 700;
    line-height: 1;
    letter-spacing: -1px;
    color: #fff;
    margin: 0 0 12px;
}
.hero-title span { color: var(--red); }
.hero-sub {
    font-family: var(--body);
    font-size: 15px;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 28px;
}
.hero-stat {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
}
.hero-stat b {
    display: block;
    font-size: 22px;
    color: var(--red);
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.2;
}

/* ── Main layout ── */
.cm-layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    min-height: calc(100vh - 180px);
}

/* ── Left panel ── */
.cm-panel-left {
    background: var(--bg2);
    border-right: 1px solid var(--border-dim);
    padding: 32px 28px;
    position: relative;
}
.cm-panel-left::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(to bottom, var(--red), transparent);
}

/* ── Right panel ── */
.cm-panel-right {
    background: var(--bg);
    padding: 36px 48px;
}

/* ── Section labels ── */
.cm-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cm-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.cm-section-title {
    font-family: var(--display);
    font-size: 22px;
    font-weight: 600;
    color: #fff;
    margin: 0 0 20px;
    letter-spacing: 0.3px;
}

/* ── Streamlit widgets overrides ── */
.stFileUploader > div {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 0 !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: var(--red) !important;
    background: rgba(232,19,42,0.04) !important;
}
.stFileUploader label {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--red) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--display) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    position: relative;
    overflow: hidden;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
}
.stButton > button:hover {
    background: #ff1a33 !important;
    box-shadow: 0 0 24px var(--red-glow) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #2a1a1e !important;
    color: #555 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 0 !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
    font-size: 15px !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 1px var(--red-glow) !important;
}
.stTextArea label {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: var(--red) !important;
}
.stSlider > div > div > div > div {
    background: var(--red) !important;
    border-color: var(--red) !important;
    box-shadow: 0 0 8px var(--red-glow) !important;
}
.stSlider label {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 60px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 14px 24px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--red) !important;
    border-bottom-color: var(--red) !important;
    background: transparent !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--red-dim), var(--red)) !important;
}
.stProgress > div > div {
    background: var(--bg3) !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    padding: 28px 32px;
    margin: 20px 0;
    position: relative;
    font-size: 15px;
    line-height: 1.8;
}
.answer-card::before {
    content: 'RESPONSE';
    position: absolute;
    top: -10px; left: 20px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    color: var(--red);
    background: var(--bg2);
    padding: 0 8px;
}

/* ── Citation pill ── */
.cite-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(232,19,42,0.1);
    border: 1px solid var(--border);
    padding: 3px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--red);
    margin: 4px 4px 4px 0;
    clip-path: polygon(0 0, calc(100% - 6px) 0, 100% 6px, 100% 100%, 6px 100%, 0 calc(100% - 6px));
}

/* ── Expanders (source chunks) ── */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--muted) !important;
    letter-spacing: 0.5px !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.streamlit-expanderContent {
    background: var(--bg3) !important;
    border: 1px solid var(--border-dim) !important;
    border-top: none !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: var(--muted) !important;
}

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
}
.stSuccess { border-left: 3px solid #00e676 !important; background: rgba(0,230,118,0.07) !important; }
.stError   { border-left: 3px solid var(--red) !important; background: rgba(232,19,42,0.07) !important; }
.stWarning { border-left: 3px solid #ffab00 !important; background: rgba(255,171,0,0.07) !important; }
.stInfo    { border-left: 3px solid #2979ff !important; background: rgba(41,121,255,0.07) !important; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-dim) !important;
    margin: 28px 0 !important;
}

/* ── Ingested file tags ── */
.file-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(232,19,42,0.08);
    border: 1px solid var(--border);
    padding: 4px 12px;
    font-family: var(--mono);
    font-size: 11px;
    color: #e8e8f0;
    margin: 4px 4px 4px 0;
}
.file-tag span { color: var(--red); }

/* ── Method badge ── */
.method-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--muted);
    border: 1px solid var(--border-dim);
    padding: 2px 10px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--red-dim); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
ingested_count = len(st.session_state.ingested_files)

st.markdown(f"""
<div class="citemate-hero">
    <div class="hero-badge">⬡ RAG-POWERED RESEARCH</div>
    <div class="hero-title">Cite<span>Mate</span> AI</div>
    <div class="hero-sub">Upload PDFs &nbsp;·&nbsp; Ask questions &nbsp;·&nbsp; Get answers with citations</div>
    <div class="hero-stats">
        <div class="hero-stat"><b>{ingested_count}</b>FILES INGESTED</div>
        <div class="hero-stat"><b>384</b>EMBED DIMS</div>
        <div class="hero-stat"><b>LLaMA</b>3.1 · 8B</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_main, tab_diag = st.tabs(["  ◈  RESEARCH  ", "  ◎  DIAGNOSTICS  "])


# ══════════════════════════════════════════════════════════
# TAB 1 — MAIN
# ══════════════════════════════════════════════════════════
with tab_main:
    st.markdown('<div class="cm-layout">', unsafe_allow_html=True)

    # ── LEFT PANEL — Upload ───────────────────────────────
    left, right = st.columns([1, 1.8], gap="large")

    with left:
        st.markdown('<div class="cm-panel-left">', unsafe_allow_html=True)
        st.markdown('<div class="cm-label">01 &nbsp; Upload</div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-section-title">Document Ingestion</div>', unsafe_allow_html=True)

        if st.session_state.ingested_files:
            tags = "".join(
                f'<div class="file-tag"><span>✓</span> {f}</div>'
                for f in st.session_state.ingested_files
            )
            st.markdown(tags, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "DROP PDF FILES HERE",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if uploaded_files:
            files_data = [(uf.name, uf.read()) for uf in uploaded_files]
            new_files  = [f for f in files_data if f[0] not in st.session_state.ingested_files]

            if not new_files:
                st.info("All files already ingested.")
            else:
                if st.button(f"⬡  INGEST  {len(new_files)}  PDF(S)"):
                    for filename, file_bytes in new_files:
                        if already_ingested(filename):
                            st.info(f"Already in DB: {filename}")
                            st.session_state.ingested_files.add(filename)
                            continue
                        with st.spinner(f"Processing {filename}..."):
                            try:
                                n = ingest_pdf(file_bytes, filename)
                                st.session_state.ingested_files.add(filename)
                                st.success(f"✓ {filename} — {n} chunks")
                            except Exception as e:
                                st.error(f"✗ {filename}: {e}")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT PANEL — Ask & Answer ────────────────────────
    with right:
        st.markdown('<div class="cm-panel-right">', unsafe_allow_html=True)
        st.markdown('<div class="cm-label">02 &nbsp; Query</div>', unsafe_allow_html=True)
        st.markdown('<div class="cm-section-title">Ask a Question</div>', unsafe_allow_html=True)

        question = st.text_area(
            "YOUR QUESTION",
            placeholder="What methodology does the paper use?\nWhat are the key findings?\nSummarize the conclusion...",
            height=130,
        )

        col_slider, col_btn = st.columns([2, 1], gap="medium")
        with col_slider:
            top_k = st.slider("CHUNKS TO RETRIEVE", min_value=3, max_value=10, value=5)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("⬡  GET ANSWER", disabled=not question.strip())

        if run:
            with st.spinner("Searching vector store..."):
                try:
                    chunks, method = retrieve(question, match_count=top_k)
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    st.stop()

            if not chunks:
                st.error("No content found. Check the Diagnostics tab.")
                st.stop()

            context = build_context(chunks)

            with st.spinner("Generating answer with LLaMA..."):
                try:
                    answer = ask_groq(context, question)
                except Exception as e:
                    st.error(f"LLM error: {e}")
                    st.stop()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="cm-label">03 &nbsp; Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="method-badge">retrieved via {method} &nbsp;·&nbsp; {len(chunks)} chunks</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="cm-label">04 &nbsp; Sources</div>', unsafe_allow_html=True)
            st.markdown('<div class="cm-section-title">Citation Chunks</div>', unsafe_allow_html=True)

            for i, c in enumerate(chunks, start=1):
                sim_pct = round(c.get("similarity", 0) * 100, 1)
                pills = f'<span class="cite-pill">⬡ {c["source_file"]}</span><span class="cite-pill">pg {c["page_number"]}</span><span class="cite-pill">{sim_pct}% match</span>'
                st.markdown(pills, unsafe_allow_html=True)
                with st.expander(f"[{i}]  {c['source_file']}  ·  page {c['page_number']}  ·  {sim_pct}%"):
                    st.write(c["content"])

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4, gap="small")

    with d1:
        st.markdown('<div class="cm-label">Check 01</div>', unsafe_allow_html=True)
        st.markdown("**Row Count**")
        if st.button("COUNT ROWS"):
            try:
                rows = supabase.table("documents").select("id, source_file, page_number, content").execute().data or []
                if not rows:
                    st.error("Table EMPTY")
                else:
                    st.success(f"{len(rows)} chunks found")
                    counts = Counter(r["source_file"] for r in rows)
                    for fname, n in counts.items():
                        st.write(f"• **{fname}** — {n}")
            except Exception as e:
                st.error(str(e))

    with d2:
        st.markdown('<div class="cm-label">Check 02</div>', unsafe_allow_html=True)
        st.markdown("**Embeddings**")
        if st.button("INSPECT EMBED"):
            try:
                rows = supabase.table("documents").select("id, source_file, embedding").limit(3).execute().data or []
                for r in rows:
                    emb = r.get("embedding")
                    if emb is None:
                        st.error(f"{r['source_file']}: NULL embedding!")
                    elif isinstance(emb, (str, list)):
                        vals = emb.strip("[]").split(",") if isinstance(emb, str) else emb
                        st.success(f"{r['source_file']}: {len(vals)} dims ✓")
            except Exception as e:
                st.error(str(e))

    with d3:
        st.markdown('<div class="cm-label">Check 03</div>', unsafe_allow_html=True)
        st.markdown("**RPC Function**")
        if st.button("TEST RPC"):
            try:
                test_vec = vec_to_string(embed(["explain the paper"])[0])
                result   = supabase.rpc("match_documents", {"query_embedding": test_vec, "match_count": 3}).execute()
                rows     = result.data or []
                if rows:
                    st.success(f"RPC: {len(rows)} rows ✓")
                    for r in rows:
                        st.write(f"• {r['source_file']} p{r['page_number']} {round(r['similarity'],3)}")
                else:
                    st.warning("RPC returned 0 rows")
                    st.code("DROP INDEX IF EXISTS idx_embedding;\nCREATE INDEX idx_embedding ON public.documents\nUSING hnsw (embedding vector_cosine_ops);", language="sql")
            except Exception as e:
                st.error(str(e))

    with d4:
        st.markdown('<div class="cm-label">Check 04</div>', unsafe_allow_html=True)
        st.markdown("**Python Fallback**")
        if st.button("TEST FALLBACK"):
            try:
                vec    = embed(["explain the paper"])[0]
                chunks = retrieve_fallback(vec, match_count=3)
                if chunks:
                    st.success(f"Fallback: {len(chunks)} chunks ✓")
                    for c in chunks:
                        st.write(f"• {c['source_file']} p{c['page_number']} {round(c['similarity'],3)}")
                else:
                    st.error("Fallback empty — table has no data")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.markdown('<div class="cm-label">Reset</div>', unsafe_allow_html=True)
    if st.button("⬡  DELETE ALL ROWS  (RESET DATABASE)", type="secondary"):
        try:
            supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            st.session_state.ingested_files = set()
            st.success("Database cleared. Session reset.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
