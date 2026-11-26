from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persist_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "In what year did tesla bigin production of the Roadster"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")

print("-----------Context----------")

for i, docs in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n {docs.page_content} \n")
