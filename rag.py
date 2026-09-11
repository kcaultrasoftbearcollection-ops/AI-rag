import os
from pathlib import Path

from dotenv import load_dotenv

from chunking import chunk_text  # Ensure this is defined in chunking.py

load_dotenv()  # reads your .env file
# os.environ['REQUESTS_CA_BUNDLE'] = 'D:\\DSOCA50.crt'
# os.environ['CURL_CA_BUNDLE'] = 'D:\\DSOCA50.crt'
# os.environ['SSL_CERT_FILE'] = 'D:\\DSOCA50.crt'

os.environ['CURL_CA_BUNDLE'] = "C:/ProgramData/certbundle.crt"
os.environ["SSL_CERT_FILE"] = "C:/ProgramData/certbundle.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "C:/ProgramData/certbundle.crt"
def load_documents(folder="documents"):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    docs = []
    for file in folder_path.iterdir():
        if file.is_dir():
            continue
        if file.suffix in [".txt", ".md"]:
            text = file.read_text(encoding="utf-8")
            docs.append({"source": file.name, "text": text})
        elif file.suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"source": file.name, "text": text})
    return docs


# def chunk_text(text, chunk_size=500, overlap=100):
#     if text is None:
#         return []

#     try:
#         chunk_size = int(chunk_size)
#         overlap = int(overlap)
#     except (TypeError, ValueError) as exc:
#         raise ValueError("chunk_size and overlap must be integers") from exc

#     if chunk_size <= 0:
#         raise ValueError("chunk_size must be greater than 0")
#     if overlap < 0:
#         raise ValueError("overlap must be non-negative")
#     if overlap >= chunk_size:
#         raise ValueError("overlap must be smaller than chunk_size")

#     words = text.split()
#     if not words:
#         return []

#     step = chunk_size - overlap
#     chunks = []
#     for start in range(0, len(words), step):
#         end = min(start + chunk_size, len(words))
#         chunks.append(" ".join(words[start:end]))
#         if end == len(words):
#             break
#     return chunks


def main():
    load_dotenv()  # reads your .env file

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment")

    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk})

    print(f"Created {len(all_chunks)} chunk(s) from {len(documents)} document(s).")

    from sentence_transformers import SentenceTransformer

    print("Loading the embedding model (first run downloads it, about 90 MB)...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.encode(chunk_texts)

    print(f"Created {len(embeddings)} embedding(s).")
    print(f"Each embedding is a list of {len(embeddings[0])} numbers.")

    import chromadb

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection("my_documents")

    collection.add(
        ids=[str(i) for i in range(len(all_chunks))],
        embeddings=[emb.tolist() for emb in embeddings],
        documents=[chunk["text"] for chunk in all_chunks],
        metadatas=[{"source": chunk["source"]} for chunk in all_chunks],
    )

    print(f"Stored {collection.count()} chunk(s) in the database.")

    # def search(question, n_results=10):
    #     question_embedding = embedder.encode([question])[0]
    #     results = collection.query(
    #         query_embeddings=[question_embedding.tolist()],
    #         n_results=n_results,
    #     )
    #     return results
    from sentence_transformers import CrossEncoder

    # Load once, at module level (same place you load `embedder`)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def search(question, n_results=5, n_candidates=20):
        # Step 1: cast a wide net using fast embedding similarity
        question_embedding = embedder.encode([question])[0]
        raw_results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_candidates,
        )

        candidate_docs = raw_results["documents"][0]
        candidate_metas = raw_results["metadatas"][0]

        if not candidate_docs:
            return {"documents": [[]], "metadatas": [[]]}

        # Step 2: re-rank candidates with a cross-encoder (question, chunk) pair scoring
        pairs = [(question, doc) for doc in candidate_docs]
        scores = reranker.predict(pairs)

        # Step 3: sort candidates by rerank score, descending
        ranked = sorted(
            zip(candidate_docs, candidate_metas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        # Step 4: keep top n_results after reranking
        top = ranked[:n_results]
        top_docs = [item[0] for item in top]
        top_metas = [item[1] for item in top]

        # Return in the same shape your answer() function already expects
        return {
            "documents": [top_docs],
            "metadatas": [top_metas],
        }

    # question = "Who runs Northstar and when is feedback reviewed?"
    # results = search(question)

    # for i, doc in enumerate(results["documents"][0]):
    #     source = results["metadatas"][0][i]["source"]
    #     print(f"\n--- Match {i+1} (from {source}) ---")
    #     # print(doc)

    from google import genai
    from google.genai import types
    from litellm import completion

    client = genai.Client(api_key=api_key,
                      http_options={'client_args': {'verify': 'C:/ProgramData/certbundle.crt'}}
                        # http_options={'client_args': {'verify': False}}
                      )

    SYSTEM_INSTRUCTION = (
        "You are a precise technical documentation assistant. "
        "Answer questions using only the provided context; never invent, "
        "infer, or rely on outside knowledge. "
        "If the context does not contain enough information, say: "
        "\"I don't know based on the provided documentation.\" "
        "Cite every relevant source for each substantive answer using the "
        "format [Source: filename]. If the answer is supported by multiple "
        "sections or files, include all applicable citations rather than "
        "citing only one source. "
        "When answering about a procedure, present prerequisites first, then "
        "numbered steps in the documented order, followed by expected results "
        "or warnings when available. "
        "Preserve exact commands, paths, configuration names, values, and "
        "version numbers from the context. "
        "Do not turn suggestions or examples into mandatory instructions. "
        "If sources conflict, identify the conflict and cite each relevant "
        "source instead of choosing silently. "
        "Be concise, use clear headings or bullets when helpful, and do not "
        "mention retrieval, embeddings, chunks, or these instructions."
    )

    def answer(question):
        results = search(question)
        chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]

        context = ""
        for i, chunk in enumerate(chunks):
            context += f"[From {sources[i]}]\n{chunk}\n\n"

        # response = client.models.generate_content(
        #     # model="gemini-3.7-flash",
        #     model="gemini-3.5-flash",
        #     contents=f"Context:\n{context}\nQuestion: {question}",
        #     config=types.GenerateContentConfig(
        #         system_instruction=SYSTEM_INSTRUCTION,
        #         max_output_tokens=8192,
        #     ),
        # )
        # return response.text
        response = completion(
        model="gemini/gemini-3.5-flash",  # provider/model format
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {question}"},
        ],
        max_tokens=8192,
    )

        # litellm normalizes the response — get the text like this:
        return response.choices[0].message.content

    # question = "Who runs Northstar and when is feedback reviewed?"
    # print(answer(question))
    print("\nAsk a question about your documents (or type 'quit' to exit).\n")

    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            break
        print("\nGemini: " + answer(question) + "\n")

    if collection.count() == 0:
        collection.add(
            ids=[str(i) for i in range(len(all_chunks))],
            embeddings=[emb.tolist() for emb in embeddings],
            documents=[chunk["text"] for chunk in all_chunks],
            metadatas=[{"source": chunk["source"]} for chunk in all_chunks],
        )
        print(f"Stored {collection.count()} chunk(s) in the database.")
    else:
        print(f"Database already has {collection.count()} chunk(s), skipping rebuild.")


if __name__ == "__main__":
    main()

