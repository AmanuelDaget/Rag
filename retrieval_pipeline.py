from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load .env file to fetch GOOGLE_API_KEY
load_dotenv()

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

# Generate final answer using LLM

combined_input = f"""Based on the following documents, please ansewer these question: {query}
      Documents:
        {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
     Please privide a clear, helpful answer using only the information from these documents. 
     If you can't find the answer in the provided document, say " I don't have enough information to answer that question".

"""

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="you are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("------Generated Response----------")
print(result.content)
