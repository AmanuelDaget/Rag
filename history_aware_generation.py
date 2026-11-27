from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
)

primary = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
backup = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model = primary.with_retry(stop_after_attempt=4).with_fallbacks([backup])

chat_history = []


def ask_question(user_question):
    print(f"You asked ----- {user_question}-------")

    if chat_history:
        messages = [
            SystemMessage(content="""Given the chat history, rewrite the new question to be standalone
             and searchable. Just return the rewritten question."""),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.text.strip()
        print(f"You are searching for: {search_question}")

    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(user_question)

    for i, doc in enumerate(relevant_docs, 1):
        print(
            f"\n[chunk {i}] {doc.metadata.get('source', '?')}\n{doc.page_content[:300]}...")
    docs_text = "\n".join(f"- {doc.page_content}" for doc in relevant_docs)

    combined_input = (
        f"Based on the following documents, answer this question: {user_question}\n\n"
        f"Documents:\n{docs_text}\n\n"
        "Please provide a clear, helpful answer using only the information from these documents. "
        "If you can't find the answer in the documents, say "
        "'I don't have enough information to answer this question'."
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",   # try 3.8-flash or 3.1-flash-lite if this one fails
        temperature=0,
    )
    message = [
        SystemMessage(content="you are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(message)
    answer = result.text

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("-----------Generated response--------")
    print(f"Answer: {answer}")

    return answer


def start_chat():
    print("Ask me question, type 'quit' to exit ")

    while True:
        question = input("\nYour question: ")

        if question.lower() == 'quit':
            print("Goodbye")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()
