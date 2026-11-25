import os
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv("Hello")

# Loading Document


def load_documents(doc_path="docs"):
    print(f"Loading all text files{doc_path}")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory {doc_path} Not exist.")

    loader = DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file exists in {doc_path} directory")

    # show the first 2 documents

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")

    return documents

# Chunking


def split_doc(documents, chunk_size=800, chunk_overlap=0):
    print("Spliting document into smaller chunks")

    text_spliter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_spliter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i+1}")
            print(f"Source metadata: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content: ")
            print(chunk.page_content)
            print("-"*50)

    return chunks

# Vector Embedings


def create_vector_store(chunk, persist_directory="db/chroma_db"):
    print("Creating embeding and storing in chromaDB")

    embeding_model = OpenAIEmbeddings(model="text-embeding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embeddings=embeding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Vector embedding finished")

    return vectorstore


def main():
    print("Hello")

    # Load documents
    documents = load_documents(doc_path="docs")

    # Chunking
    chunks = split_doc(documents)

    # Vector embeddings
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
