# CiteMate AI — Setup & Deployment Guide

## Project Structure
```
citeMate/
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
└── supabase_setup.sql
```

---

## Step 1 — Supabase Setup

1. Go to [supabase.com](https://supabase.com) and create a free project.
2. Navigate to **SQL Editor** in the sidebar.
3. Paste the entire contents of `supabase_setup.sql` and click **Run**.
4. Go to **Project Settings → API** and copy:
   - `Project URL` → this is your `SUPABASE_URL`
   - `anon public` key → this is your `SUPABASE_KEY`

---

## Step 2 — Groq API Key

1. Go to [console.groq.com](https://console.groq.com) and sign up.
2. Create an API key under **API Keys**.
3. Copy it — this is your `GROQ_API_KEY`.

---

## Step 3 — Local Development

### Create a `.env` file in the project root:
```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your_anon_key_here
GROQ_API_KEY=your_groq_key_here
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run locally:
```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## Step 4 — Deploy to Render

1. Push your project to a GitHub repository (do **not** commit `.env`).

2. Go to [render.com](https://render.com) and click **New → Web Service**.

3. Connect your GitHub repo.

4. Configure the service:
   | Setting          | Value                        |
   |------------------|------------------------------|
   | **Runtime**      | Python 3                     |
   | **Build Command**| `pip install -r requirements.txt` |
   | **Start Command**| `gunicorn app:app`           |

5. Add Environment Variables under **Environment**:
   - `SUPABASE_URL` = your Supabase project URL
   - `SUPABASE_KEY` = your Supabase anon key
   - `GROQ_API_KEY` = your Groq API key

6. Click **Deploy**. Render will build and launch your app.

> ⚠️ **Note on first deploy**: `sentence-transformers` will download the
> `all-MiniLM-L6-v2` model (~90MB) on first startup. This is normal.
> Render caches it between deploys.

---

## How It Works

```
PDF Upload  →  PyMuPDF extract text  →  Split into 600-word chunks
           →  Embed each chunk (MiniLM)  →  Store in Supabase (pgvector)

Question   →  Embed question (MiniLM)  →  Cosine similarity search (top 5)
           →  Combine chunks as context  →  Groq LLM answers with citations
           →  Return answer + source references
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `vector extension not found` | Run `CREATE EXTENSION vector;` in Supabase SQL Editor |
| `function match_documents not found` | Re-run the full `supabase_setup.sql` |
| Slow first response | MiniLM model loading on cold start; subsequent calls are fast |
| `GROQ_API_KEY not set` | Check `.env` file or Render environment variables |
