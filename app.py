import streamlit as st
import fitz
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def vec_to_string(vec):
    return "[" + ",".join(str(round(float(v), 8)) for v in vec) + "]"

def embed(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

def chunk_text(text, max_words=600):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words) if words[i:i+max_words]]

def already_ingested(filename):
    r = supabase.table("documents").select("id").eq("source_file", filename).limit(1).execute()
    return len(r.data) > 0

def ingest_pdf(file_bytes, filename):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_number": i, "text": text})
    doc.close()
    all_chunks = [{"page_number": p["page_number"], "content": c} for p in pages for c in chunk_text(p["text"])]
    if not all_chunks:
        raise ValueError("No extractable text found.")
    total = len(all_chunks)
    progress = st.progress(0, text="Processing...")
    for idx, chunk in enumerate(all_chunks):
        vec_str = vec_to_string(embed([chunk["content"]])[0])
        supabase.table("documents").insert({
            "content": chunk["content"], "embedding": vec_str,
            "page_number": chunk["page_number"], "source_file": filename,
        }).execute()
        progress.progress((idx + 1) / total, text=f"Chunk {idx+1}/{total}")
    progress.empty()
    return total

def retrieve_rpc(vec_str, match_count):
    r = supabase.rpc("match_documents", {"query_embedding": vec_str, "match_count": match_count}).execute()
    return r.data or []

def retrieve_fallback(vec, match_count):
    rows = supabase.table("documents").select("content, page_number, source_file, embedding").execute().data or []
    q = np.array(vec, dtype="float32")
    scored = []
    for row in rows:
        emb = row.get("embedding")
        if emb is None: continue
        if isinstance(emb, str): emb = [float(x) for x in emb.strip("[]").split(",")]
        scored.append({**row, "similarity": float(np.dot(q, np.array(emb, dtype="float32")))})
    return sorted(scored, key=lambda x: x["similarity"], reverse=True)[:match_count]

def retrieve(question, match_count=5):
    vec = embed([question])[0]
    vec_str = vec_to_string(vec)
    try:
        chunks = retrieve_rpc(vec_str, match_count)
        if chunks: return chunks, "RPC"
    except Exception:
        pass
    return retrieve_fallback(vec, match_count), "Fallback"

def build_context(chunks, word_limit=1500):
    parts, total = [], 0
    for c in chunks:
        words = c["content"].split()
        if total + len(words) > word_limit:
            parts.append(" ".join(words[:word_limit - total])); break
        parts.append(c["content"]); total += len(words)
    return "\n\n---\n\n".join(parts)

CITATION_PROMPTS = {
    "APA": (
        "You are a research assistant. Answer ONLY using the provided context. "
        "Format all in-text citations in APA 7th edition style: (Author, Year, p. X) or (Filename, p. X) if no author is known. "
        "At the end of your answer, add a References section listing each source in APA format. "
        "Use the filename and page number where full details are unavailable."
    ),
    "MLA": (
        "You are a research assistant. Answer ONLY using the provided context. "
        "Format all in-text citations in MLA 9th edition style: (Author Page) or (Filename Page) if no author is known. "
        "At the end of your answer, add a Works Cited section in MLA format. "
        "Use filename and page number where full details are unavailable."
    ),
    "Chicago": (
        "You are a research assistant. Answer ONLY using the provided context. "
        "Format all citations as Chicago footnote-style, indicated inline like [1], [2], etc. "
        "At the end of your answer, add a numbered Notes section with full Chicago citations. "
        "Use filename and page number where full details are unavailable."
    ),
    "Harvard": (
        "You are a research assistant. Answer ONLY using the provided context. "
        "Format all in-text citations in Harvard style: (Author Year, p. X) or (Filename Year, p. X) if no author is known. "
        "At the end of your answer, add a Reference List section in Harvard format. "
        "Use filename and page number where full details are unavailable."
    ),
    "IEEE": (
        "You are a research assistant. Answer ONLY using the provided context. "
        "Format all citations as IEEE numbered references inline like [1], [2], etc. "
        "At the end of your answer, add a numbered References section in IEEE format: "
        "[1] A. Author, Title, Publisher, Year, p. X. Use filename and page number where full details are unavailable."
    ),
}

def ask_groq(context, question, citation_format="APA"):
    system_prompt = CITATION_PROMPTS.get(citation_format, CITATION_PROMPTS["APA"])
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        temperature=0.2, max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CiteMate AI", page_icon="⬡", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# INJECT CSS — target every Streamlit wrapper aggressively
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
  --red:   #e8132a;
  --red2:  #a50e1e;
  --glow:  rgba(232,19,42,0.3);
  --bg:    #0b0b0e;
  --bg2:   #101015;
  --bg3:   #16161d;
  --bdr:   rgba(232,19,42,0.2);
  --bdr2:  rgba(255,255,255,0.05);
  --txt:   #eeeef5;
  --muted: #6b6b82;
}

/* ─── Nuclear dark override ─── */
html, body { background: var(--bg) !important; }

/* Kill every Streamlit white container */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] > section,
[data-testid="block-container"],
.main, .main > div,
div[class*="appview"],
div[class*="main"],
section[class*="main"] {
  background: var(--bg) !important;
  background-color: var(--bg) !important;
}

/* Every generic div/section gets dark bg */
.stMarkdown, .element-container, .stVerticalBlock,
.stHorizontalBlock, div[data-testid*="column"] {
  background: transparent !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}

[data-testid="block-container"] {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ─── Animated dot-grid bg ─── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: radial-gradient(rgba(232,19,42,0.12) 1px, transparent 1px);
  background-size: 30px 30px;
  pointer-events: none;
  z-index: 0;
}

/* ─── Tabs ─── */
[data-testid="stTabs"],
[data-baseweb="tab-list"],
div[role="tablist"] {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--bdr) !important;
  padding: 0 40px !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  padding: 16px 28px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--red) !important;
  border-bottom: 2px solid var(--red) !important;
  background: transparent !important;
}
[data-baseweb="tab-panel"] {
  background: var(--bg) !important;
  padding: 0 !important;
}

/* ─── Buttons ─── */
.stButton > button {
  background: var(--red) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 0 !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  padding: 11px 24px !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 0 0 transparent !important;
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px)) !important;
}
.stButton > button:hover {
  background: #ff1530 !important;
  box-shadow: 0 0 30px var(--glow) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:disabled {
  background: #1e1015 !important;
  color: #3a3a4a !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ─── Text area ─── */
.stTextArea > div > div {
  background: var(--bg3) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 0 !important;
}
.stTextArea textarea {
  background: var(--bg3) !important;
  color: var(--txt) !important;
  font-family: 'Exo 2', sans-serif !important;
  font-size: 15px !important;
  border: none !important;
  caret-color: var(--red) !important;
}
.stTextArea textarea:focus {
  outline: none !important;
  box-shadow: none !important;
}
.stTextArea > div > div:focus-within {
  border-color: var(--red) !important;
  box-shadow: 0 0 0 1px var(--glow) !important;
}
.stTextArea label, .stFileUploader label, .stSlider label {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}

/* ─── File uploader ─── */
[data-testid="stFileUploader"] > div {
  background: var(--bg3) !important;
  border: 1px dashed var(--bdr) !important;
  border-radius: 0 !important;
  transition: all 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
  border-color: var(--red) !important;
  background: rgba(232,19,42,0.05) !important;
}
[data-testid="stFileUploader"] * { color: var(--muted) !important; }

/* ─── Slider ─── */
[data-testid="stSlider"] > div > div > div {
  background: var(--red) !important;
}
[data-testid="stSlider"] > div > div > div > div {
  background: var(--red) !important;
  box-shadow: 0 0 10px var(--glow) !important;
}

/* ─── Expanders ─── */
[data-testid="stExpander"] {
  background: var(--bg3) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 0 !important;
  margin-bottom: 6px !important;
}
[data-testid="stExpander"] > div > div {
  background: var(--bg3) !important;
}
[data-testid="stExpander"] summary {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 11px !important;
  color: var(--muted) !important;
  background: var(--bg3) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--txt) !important; }

/* ─── Alerts ─── */
[data-testid="stAlert"] {
  border-radius: 0 !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 12px !important;
}
.stSuccess { background: rgba(0,230,118,0.06) !important; border-left: 2px solid #00e676 !important; }
.stError   { background: rgba(232,19,42,0.08) !important; border-left: 2px solid var(--red) !important; }
.stWarning { background: rgba(255,171,0,0.06) !important; border-left: 2px solid #ffab00 !important; }
.stInfo    { background: rgba(41,121,255,0.06) !important; border-left: 2px solid #2979ff !important; }

/* ─── Progress bar ─── */
[data-testid="stProgressBar"] > div {
  background: var(--bg3) !important;
  border-radius: 0 !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--red2), var(--red)) !important;
  border-radius: 0 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--red2); }

/* ─── Custom components ─── */
.hero-wrap {
  background: linear-gradient(135deg, #0d0d12 0%, #120508 60%, #0d0d12 100%);
  padding: 50px 56px 42px;
  border-bottom: 1px solid var(--bdr);
  position: relative;
  overflow: hidden;
}
.hero-wrap::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--red) 40%, var(--red) 60%, transparent 100%);
  animation: pulse-line 3s ease-in-out infinite;
}
@keyframes pulse-line { 0%,100%{opacity:.3} 50%{opacity:1} }

.hero-glow {
  position: absolute;
  top: -80px; right: -80px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(232,19,42,0.15), transparent 65%);
  pointer-events: none;
}
.hero-tag {
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  letter-spacing: 4px;
  color: var(--red);
  border: 1px solid var(--bdr);
  background: rgba(232,19,42,0.08);
  padding: 4px 14px;
  margin-bottom: 16px;
  text-transform: uppercase;
}
.hero-h1 {
  font-family: 'Rajdhani', sans-serif;
  font-size: clamp(44px, 6vw, 80px);
  font-weight: 700;
  color: #ffffff;
  line-height: 0.95;
  letter-spacing: -2px;
  margin: 0 0 10px;
}
.hero-h1 em { color: var(--red); font-style: normal; }
.hero-desc {
  font-family: 'Exo 2', sans-serif;
  font-size: 14px;
  font-weight: 300;
  color: var(--muted);
  letter-spacing: 1px;
  margin-top: 8px;
}
.stats-row { display: flex; gap: 40px; margin-top: 30px; }
.stat-item { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 2px; color: var(--muted); }
.stat-item strong { display: block; font-size: 28px; color: var(--red); font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: -1px; line-height: 1; }

.panel {
  background: var(--bg2);
  border: 1px solid var(--bdr2);
  padding: 32px 28px;
  position: relative;
  height: 100%;
}
.panel-left { border-right: 1px solid var(--bdr2); }
.panel-left::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 2px; height: 60%;
  background: linear-gradient(to bottom, var(--red), transparent);
}

.sec-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  letter-spacing: 3px;
  color: var(--red);
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.sec-label::after { content:''; flex:1; height:1px; background:var(--bdr); }

.sec-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 24px;
  font-weight: 600;
  color: #ffffff;
  margin: 0 0 20px;
  letter-spacing: 0.5px;
}

.file-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(232,19,42,0.1);
  border: 1px solid var(--bdr);
  padding: 4px 12px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px;
  color: #eeeef5;
  margin: 0 4px 6px 0;
}
.file-chip b { color: var(--red); }

.answer-box {
  background: var(--bg3);
  border: 1px solid var(--bdr);
  border-left: 3px solid var(--red);
  padding: 28px 32px;
  margin: 12px 0 24px;
  font-family: 'Exo 2', sans-serif;
  font-size: 15px;
  line-height: 1.85;
  color: var(--txt);
  position: relative;
}
.answer-box::before {
  content: 'AI RESPONSE';
  position: absolute;
  top: -9px; left: 18px;
  font-family: 'Share Tech Mono', monospace;
  font-size: 9px;
  letter-spacing: 3px;
  color: var(--red);
  background: var(--bg3);
  padding: 0 8px;
}

.cite-tag {
  display: inline-block;
  background: rgba(232,19,42,0.1);
  border: 1px solid var(--bdr);
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  color: var(--red);
  padding: 2px 10px;
  margin: 3px 3px 3px 0;
}

.method-tag {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--muted);
  border: 1px solid var(--bdr2);
  padding: 3px 12px;
  display: inline-block;
  margin-bottom: 12px;
}


/* ─── Citation format selector ─── */
div[data-testid="stRadio"] > label {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 3px !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
div[data-testid="stRadio"] > div {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 8px !important;
  margin-top: 8px !important;
}
div[data-testid="stRadio"] > div > label {
  background: var(--bg3) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 0 !important;
  padding: 7px 16px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 2px !important;
  color: var(--muted) !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
  clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px)) !important;
}
div[data-testid="stRadio"] > div > label:hover {
  border-color: var(--bdr) !important;
  color: var(--txt) !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
  background: rgba(232,19,42,0.12) !important;
  border-color: var(--red) !important;
  color: var(--red) !important;
}
div[data-testid="stRadio"] input { display: none !important; }

.cite-format-box {
  background: var(--bg3);
  border: 1px solid var(--bdr2);
  border-top: 2px solid var(--red);
  padding: 18px 20px 20px;
  margin-bottom: 16px;
}
.cite-format-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 9px;
  letter-spacing: 3px;
  color: var(--red);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.cite-format-desc {
  font-family: 'Exo 2', sans-serif;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 12px;
}

.diag-card {
  background: var(--bg2);
  border: 1px solid var(--bdr2);
  border-top: 2px solid var(--red);
  padding: 24px 20px;
}
.diag-title {
  font-family: 'Share Tech Mono', monospace;
  font-size: 10px;
  letter-spacing: 3px;
  color: var(--red);
  text-transform: uppercase;
  margin-bottom: 6px;
}
.diag-desc {
  font-family: 'Exo 2', sans-serif;
  font-size: 13px;
  color: var(--muted);
  margin-bottom: 16px;
  line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
n_files = len(st.session_state.ingested_files)
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-tag">⬡ &nbsp; RAG-POWERED RESEARCH ASSISTANT</div>
  <div class="hero-h1">Cite<em>Mate</em><br>AI</div>
  <div class="hero-desc">UPLOAD PDFS &nbsp;·&nbsp; SEMANTIC SEARCH &nbsp;·&nbsp; CITED ANSWERS</div>
  <div class="stats-row">
    <div class="stat-item"><strong>{n_files}</strong>DOCS LOADED</div>
    <div class="stat-item"><strong>384</strong>EMBED DIMS</div>
    <div class="stat-item"><strong>8B</strong>LLaMA MODEL</div>
    <div class="stat-item"><strong>RAG</strong>PIPELINE</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_main, tab_diag = st.tabs(["  ◈  RESEARCH  ", "  ◎  DIAGNOSTICS  "])


# ══════════════════════════════════════════════════════════════════════
# RESEARCH TAB
# ══════════════════════════════════════════════════════════════════════
with tab_main:
    col_left, col_right = st.columns([1, 1.7], gap="small")

    # ── LEFT: UPLOAD ──────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="panel panel-left">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">01 &nbsp; Upload</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Document Ingestion</div>', unsafe_allow_html=True)

        if st.session_state.ingested_files:
            chips = "".join(f'<span class="file-chip"><b>✓</b> {f}</span>' for f in st.session_state.ingested_files)
            st.markdown(chips + "<br><br>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader("DROP PDF FILES HERE", type="pdf", accept_multiple_files=True)

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
                                st.success(f"✓ {filename} · {n} chunks")
                            except Exception as e:
                                st.error(f"✗ Failed: {e}")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: ASK + ANSWER ───────────────────────────────────────────
    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">02 &nbsp; Query</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Ask a Question</div>', unsafe_allow_html=True)

        question = st.text_area(
            "YOUR QUESTION",
            placeholder="What methodology does the paper use?\nSummarize the key findings...\nWhat are the conclusions?",
            height=120,
        )

        # ── Citation format selector ──────────────────────
        FORMAT_INFO = {
            "APA":     "Author-date in-text · References list",
            "MLA":     "Author-page in-text · Works Cited",
            "Chicago": "Numbered footnotes [1] · Notes list",
            "Harvard": "Author-year in-text · Reference List",
            "IEEE":    "Numbered inline [1] · IEEE References",
        }
        st.markdown('''<div class="cite-format-box">
  <div class="cite-format-title">⬡ &nbsp; Citation Format</div>
  <div class="cite-format-desc">Choose how sources are cited in the answer</div>''', unsafe_allow_html=True)

        cite_format = st.radio(
            "CITATION FORMAT",
            options=list(FORMAT_INFO.keys()),
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div style="font-family:Share Tech Mono,monospace;font-size:10px;color:#6b6b82;letter-spacing:2px;margin-top:6px">{FORMAT_INFO[cite_format]}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1], gap="medium")
        with c1:
            top_k = st.slider("CHUNKS TO RETRIEVE", min_value=3, max_value=10, value=5)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("⬡  GET ANSWER", disabled=not question.strip())

        if run:
            with st.spinner("Searching vector store..."):
                try:
                    chunks, method = retrieve(question, match_count=top_k)
                except Exception as e:
                    st.error(f"Retrieval error: {e}"); st.stop()

            if not chunks:
                st.error("No results found. Open Diagnostics tab to investigate.")
                st.stop()

            with st.spinner("Generating answer..."):
                try:
                    answer = ask_groq(build_context(chunks), question, citation_format=cite_format)
                except Exception as e:
                    st.error(f"LLM error: {e}"); st.stop()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">03 &nbsp; Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="method-tag">via {method} &nbsp;·&nbsp; {len(chunks)} chunks &nbsp;·&nbsp; {cite_format}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-label">04 &nbsp; Sources</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Citation Chunks</div>', unsafe_allow_html=True)
            for i, c in enumerate(chunks, start=1):
                sim = round(c.get("similarity", 0) * 100, 1)
                tags = f'<span class="cite-tag">⬡ {c["source_file"]}</span><span class="cite-tag">pg {c["page_number"]}</span><span class="cite-tag">{sim}% match</span>'
                st.markdown(tags, unsafe_allow_html=True)
                with st.expander(f"[{i}]  {c['source_file']}  ·  page {c['page_number']}  ·  {sim}%"):
                    st.write(c["content"])

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS TAB
# ══════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4, gap="small")

    with d1:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown('<div class="diag-title">Check 01 · Row Count</div>', unsafe_allow_html=True)
        st.markdown('<div class="diag-desc">Count all stored chunks in Supabase</div>', unsafe_allow_html=True)
        if st.button("COUNT ROWS"):
            try:
                rows = supabase.table("documents").select("id, source_file, page_number, content").execute().data or []
                if not rows:
                    st.error("Table is EMPTY")
                else:
                    st.success(f"{len(rows)} chunks found")
                    for fname, n in Counter(r["source_file"] for r in rows).items():
                        st.write(f"• **{fname}** — {n}")
            except Exception as e:
                st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown('<div class="diag-title">Check 02 · Embeddings</div>', unsafe_allow_html=True)
        st.markdown('<div class="diag-desc">Verify embedding column is populated</div>', unsafe_allow_html=True)
        if st.button("INSPECT EMBED"):
            try:
                rows = supabase.table("documents").select("id, source_file, embedding").limit(3).execute().data or []
                for r in rows:
                    emb = r.get("embedding")
                    if emb is None:
                        st.error(f"{r['source_file']}: NULL!")
                    else:
                        vals = emb.strip("[]").split(",") if isinstance(emb, str) else emb
                        st.success(f"{r['source_file'][:20]}: {len(vals)} dims ✓")
            except Exception as e:
                st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)

    with d3:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown('<div class="diag-title">Check 03 · RPC</div>', unsafe_allow_html=True)
        st.markdown('<div class="diag-desc">Test match_documents function directly</div>', unsafe_allow_html=True)
        if st.button("TEST RPC"):
            try:
                test_vec = vec_to_string(embed(["explain the paper"])[0])
                result   = supabase.rpc("match_documents", {"query_embedding": test_vec, "match_count": 3}).execute()
                rows     = result.data or []
                if rows:
                    st.success(f"RPC OK · {len(rows)} rows")
                    for r in rows:
                        st.write(f"• {r['source_file']} p{r['page_number']}")
                else:
                    st.warning("RPC returned 0 rows")
                    st.code("DROP INDEX IF EXISTS idx_embedding;\nCREATE INDEX idx_embedding\nON public.documents\nUSING hnsw (embedding vector_cosine_ops);", language="sql")
            except Exception as e:
                st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)

    with d4:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        st.markdown('<div class="diag-title">Check 04 · Fallback</div>', unsafe_allow_html=True)
        st.markdown('<div class="diag-desc">Python cosine ranking without RPC</div>', unsafe_allow_html=True)
        if st.button("TEST FALLBACK"):
            try:
                vec    = embed(["explain the paper"])[0]
                chunks = retrieve_fallback(vec, match_count=3)
                if chunks:
                    st.success(f"Fallback OK · {len(chunks)} chunks")
                    for c in chunks:
                        st.write(f"• {c['source_file']} p{c['page_number']} {round(c['similarity'],3)}")
                else:
                    st.error("No data — table empty")
            except Exception as e:
                st.error(str(e))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">⬡ &nbsp; Danger Zone</div>', unsafe_allow_html=True)
    if st.button("⬡  DELETE ALL ROWS  ·  RESET DATABASE", type="secondary"):
        try:
            supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            st.session_state.ingested_files = set()
            st.success("Database cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
