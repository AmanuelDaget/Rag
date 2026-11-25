import os
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv("Hello")

def load_documents(doc_path = "docs"):
    print(f"Loading all text files{doc_path}")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory {doc_path} Not exist.")

    loader = DirectoryLoader(
        path= doc_path,
        glob= "*.txt",
        loader_cls= TextLoader
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file exists in {doc_path} directory")

    #show the first 2 documents

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")

    
def main():
    print("Hello")

    #Load documents
    documents = load_documents(doc_path="docs")


if __name__ == "__main__":
    main()
