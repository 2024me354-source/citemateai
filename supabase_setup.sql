-- ============================================================
-- CiteMate AI — Supabase Setup Script
-- Run this in: Supabase Dashboard → SQL Editor
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the documents table
CREATE TABLE IF NOT EXISTS documents (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content     TEXT NOT NULL,
  embedding   VECTOR(384),          -- all-MiniLM-L6-v2 produces 384-dim vectors
  page_number INTEGER,
  source_file TEXT
);

-- 3. Create IVFFlat index for fast cosine similarity search
--    (IVFFlat is best for datasets < 1M rows; lists=100 is a good default for small sets)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- 4. Create the RPC function used by the Flask app for similarity search
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(384),
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  id          UUID,
  content     TEXT,
  page_number INTEGER,
  source_file TEXT,
  similarity  FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    id,
    content,
    page_number,
    source_file,
    1 - (embedding <=> query_embedding) AS similarity
  FROM documents
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- ✅ Done! No RLS. Table is public.
-- The Flask app will now be able to insert rows and call match_documents().
