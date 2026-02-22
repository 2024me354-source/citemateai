-- Run this in your Supabase SQL editor (Dashboard → SQL Editor → New Query)

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create documents table
create table if not exists documents (
    id          uuid primary key default gen_random_uuid(),
    content     text,
    embedding   vector(384),
    page_number integer,
    source_file text
);

-- 3. Create IVFFlat index for fast cosine similarity search
create index if not exists idx_embedding
on documents
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- 4. Create similarity search RPC function
create or replace function match_documents(
  query_embedding vector(384),
  match_count     int default 5
)
returns table (
  content     text,
  page_number integer,
  source_file text,
  similarity  float
)
language sql stable
as $$
  select
    content,
    page_number,
    source_file,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
