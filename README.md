# CiteMate AI – PDF Research Assistant

RAG-powered research assistant: upload PDFs, ask questions, get cited answers.

## Stack
| Layer | Tool |
|-------|------|
| UI + backend | Streamlit |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| PDF parsing | PyMuPDF |
| Vector DB | Supabase + pgvector |
| LLM | Groq (llama3-8b-8192) |

---

## 1 · Supabase Setup

1. Create a free project at https://supabase.com
2. Go to **SQL Editor → New Query**
3. Paste and run the contents of `supabase_setup.sql`
4. Note your **Project URL** and **anon/service_role key** from *Settings → API*

---

## 2 · Local Development

```bash
# Clone / cd into project
pip install -r requirements.txt

# Create .streamlit/secrets.toml (local only – do NOT commit)
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-supabase-anon-key"
GROQ_API_KEY  = "your-groq-api-key"
EOF

streamlit run app.py
```

---

## 3 · Deploy to Streamlit Cloud

1. Push the repo to GitHub (make sure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to https://share.streamlit.io → **New app**
3. Select your repo and set **Main file path** to `app.py`
4. Open **Advanced settings → Secrets** and paste:

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "your-supabase-anon-key"
GROQ_API_KEY  = "your-groq-api-key"
```

5. Click **Deploy** – done!

---

## Usage

1. Upload one or more PDF files in the **Upload** section and click **Ingest PDFs**
2. Type a question in the **Ask** section and click **Get Answer**
3. The answer appears with inline citations `(filename, page)` and expandable source chunks below
