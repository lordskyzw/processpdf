import os
from flask import Flask, request, Response
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
import numpy as np
import pinecone
import psycopg2

app = Flask(__name__)

# Initialize OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize Pinecone with API key and environment
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="northamerica-northeast1-gcp",
)

# Initialize PostgreSQL connection
db_url = os.environ.get(
    "postgresql://postgres:emqm0bjXNhlbesD9udyQ@containers-us-west-190.railway.app:6490/railway"
)
if db_url is not None:
    conn = psycopg2.connect(db_url, sslmode="require")
else:
    conn = None

# Define table schema
table_schema = """
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    uuid TEXT NOT NULL,
    filename TEXT NOT NULL,
    data BYTEA NOT NULL
);
"""

# Create table if it does not exist
if conn is not None:
    with conn.cursor() as cur:
        cur.execute(table_schema)
        conn.commit()


@app.route("/", methods=["GET"])
def sayhi():
    return "Tarmica says welcome to the file processing engine!"


@app.route("/file_embeddings", methods=["GET"])
def default_response():
    return "You made a GET request! Make a POST request for the interesting magic!"


# Define endpoint to receive PDF and text documents and store vector embeddings in Pinecone
@app.route("/file_embeddings", methods=["POST"])
def file_embeddings():
    # Load document from request
    try:
        file = request.files["file"]
        if not file.filename.endswith(".pdf") and not file.filename.endswith(".txt"):
            return "Invalid file format. Only PDF and TXT files are allowed.", 400
        uuid = request.form["uuid"]
    except Exception as e:
        return f"Error loading document: {str(e)}", 400

    # Save file to database
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files (uuid, filename, data) VALUES (%s, %s, %s)",
                (uuid, file.filename, file.read()),
            )
            conn.commit()

    # Set response headers for streaming
    def generate():
        yield "Starting document processing...\n"

        try:
            # Preprocess text content of document
            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(file)
                doc = loader.load()
                char_text_splitter = CharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                doc_texts = char_text_splitter.split_document(doc)
            elif file.filename.endswith(".txt"):
                doc_texts = [file.read().decode("utf-8")]

            # Create vector embeddings of preprocessed text
            vector_embeddings = []
            for i, d in enumerate(doc_texts):
                embeddings = openai.Completion.create(
                    engine="text-embedding-ada-002",
                    prompt=d,
                    max_tokens=512,
                    temperature=0.5,
                )["choices"][0]["embedding"]
                vector_embeddings.append(np.array(embeddings).tolist())
                yield f"Processed page {i+1} of {len(doc_texts)}\n"

            vector_embeddings = np.concatenate(vector_embeddings, axis=0)

            # Store vector embeddings in Pinecone
            index_name = "schooldocs"

            pinecone_index = pinecone.Index(index_name)
            index_ids = pinecone_index.add_ids([file.filename], vector_embeddings)

            # Wait for index update to complete
            pinecone_index.wait_index_created()

            # Update Pinecone index metadata with UUID and finished status
            index_metadata = {"uuid": uuid, "finished": True}
            pinecone_index.upsert_metadata(index_ids[0], index_metadata)

            # Get Pinecone entry and send final response chunk
            pinecone_entry = pinecone_index.query(ids=[index_ids[0]])[0]
            yield f"Processing complete! Pinecone entry: {pinecone_entry}\n"
            yield "end"

        except Exception as e:
            yield f"Error processing document: {str(e)}\n"
            yield "end"

    return Response(generate(), mimetype="text/plain")


@app.route("/file_data/<uuid>", methods=["GET"])
def file_data(uuid):
    # Retrieve file data from database using UUID
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, data FROM files WHERE uuid = %s", (uuid,))
            result = cur.fetchone()
            if result is None:
                return "File not found.", 404
            filename, data = result
            return Response(
                data,
                mimetype="application/octet-stream",
                headers={"Content-Disposition": f"attachment;filename={filename}"},
            )
    else:
        return "Database connection not available.", 500


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
