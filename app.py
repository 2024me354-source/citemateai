import os
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

app = Flask(__name__)

# --- Clients ---
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Helpers ---

def extract_chunks_from_pdf(file_bytes, filename):
    """Extract text chunks (500-800 words) with page numbers from a PDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        words = text.split()
        # Split page text into ~600-word chunks
        chunk_size = 600
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text) > 50:  # skip tiny fragments
                chunks.append({
                    "content": chunk_text,
                    "page_number": page_num + 1,
                    "source_file": filename
                })

    doc.close()
    return chunks


def embed_text(text):
    """Generate a 384-dim embedding vector."""
    return embedder.encode(text).tolist()


def store_chunks(chunks):
    """Insert chunks with embeddings into Supabase."""
    rows = []
    for chunk in chunks:
        embedding = embed_text(chunk["content"])
        rows.append({
            "content": chunk["content"],
            "embedding": embedding,
            "page_number": chunk["page_number"],
            "source_file": chunk["source_file"]
        })
    # Insert in batches of 50
    for i in range(0, len(rows), 50):
        supabase.table("documents").insert(rows[i:i+50]).execute()


def search_similar_chunks(question, top_k=5):
    """Embed question and find top-k similar chunks via pgvector."""
    question_embedding = embed_text(question)
    result = supabase.rpc("match_documents", {
        "query_embedding": question_embedding,
        "match_count": top_k
    }).execute()
    return result.data


def ask_groq(question, context_chunks):
    """Send question + context to Groq and return answer with references."""
    context_text = ""
    references = []

    for i, chunk in enumerate(context_chunks):
        context_text += f"\n[Source {i+1}: {chunk['source_file']}, Page {chunk['page_number']}]\n{chunk['content']}\n"
        ref = f"{chunk['source_file']} (Page {chunk['page_number']})"
        if ref not in references:
            references.append(ref)

    prompt = f"""You are a helpful research assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I could not find relevant information in the uploaded documents."
Always cite the source file and page number when referencing information.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )

    answer = response.choices[0].message.content.strip()
    return answer, references


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    file_bytes = file.read()
    filename = file.filename

    try:
        chunks = extract_chunks_from_pdf(file_bytes, filename)
        if not chunks:
            return jsonify({"error": "Could not extract text from PDF"}), 400

        store_chunks(chunks)
        return jsonify({
            "message": f"Successfully indexed '{filename}'",
            "chunks_indexed": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        context_chunks = search_similar_chunks(question, top_k=5)
        if not context_chunks:
            return jsonify({"answer": "No relevant documents found. Please upload some PDFs first.", "references": []})

        answer, references = ask_groq(question, context_chunks)
        return jsonify({"answer": answer, "references": references})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
