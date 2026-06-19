from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

presist_directory = "db/chroma_db"

embedding_function = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(persist_directory=presist_directory, embedding_function=embedding_function)

model = ChatOllama(model="minicpm-v4.6")

chatHistory = []

def ask_query(query):
    retrieval_results = db.as_retriever(search_kwargs={"k": 3})
    if(len(chatHistory) == 0):
        userQuery = query
    else:
        messages = [SystemMessage(content="Given the chat history and the new user query, provide a concise summary of the conversation so far. Then, combine this summary with the new user query to create an updated query that captures the context of the conversation. The summary should be brief but informative, highlighting key points from the chat history that are relevant to the new query.")] + chatHistory + [HumanMessage(content=f"New Query: {query}")]
        result = model.invoke(messages)
        userQuery = result.content.strip()
        print(f"Updated Query with Context: {userQuery}")

    results = retrieval_results.invoke(userQuery)
    chatHistory.append(HumanMessage(content=f"User Query: {userQuery}"))

    print("Top 3 relevant chunks:")
    for i, result in enumerate(results):
        print(f"-------Result {i+1}------")
        print(f"Source: {result.metadata['source']}")
        print(f"Length: {len(result.page_content)} characters")
        print(f"Content: {result.page_content}")
        print("-"*50)

    combined_input = f"""Based on the retrieved chunks, provide a comprehensive answer to the query: "{userQuery}".
    Chunks:
    {chr(10).join([f"-{result.page_content}" for result in results])}
    Use the information from the chunks to construct a detailed response.
    If the chunks do not contain sufficient information, indicate that more information is needed.
    """

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. Use ONLY the provided context chunks to answer the user's question. "
            "If the answer cannot be found in the context, state that you do not have enough information. "
            "Do not use outside knowledge. Cite the source document if possible."
        )),
        HumanMessage(content=combined_input)
    ]
    
    response = model.invoke(messages)
    print("Response from the model:")
    print(response.content)


def startChar():
    print("Welcome to the RAG Agent!")
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the RAG Agent. Goodbye!")
            break

        ask_query(query)

if __name__ == "__main__":
    startChar()