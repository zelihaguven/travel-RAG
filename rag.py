import os
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are an expert travel assistant. Use the travel reviews and destination info below to create a detailed, personalised travel plan.

User's request:
- City: {city}
- Budget: {budget}
- Duration: {duration} days
- Travelers: {travelers}
- Interests: {interests}

Relevant reviews and local info:
{context}

Please provide a structured plan with:
1. **Hotel Recommendations** – real names from the reviews, with price range
2. **Restaurant Recommendations** – specific places, must-try dishes, estimated costs
3. **Activities & Attractions** – top picks matching the interests above
4. **Day-by-Day Itinerary** – hour-by-hour plan for each day
5. **Budget Breakdown** – estimated total spend per category
6. **Money-Saving Tips** – practical advice for this destination

Be specific and mention real place names from the reviews. Answer in a friendly, helpful tone."""

def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    count = vectorstore._collection.count()
    print(f"Vectorstore loaded: {count} vectors")
    if count == 0:
        raise RuntimeError("ChromaDB is empty. Run `python ingest.py` first.")
    return vectorstore

def create_rag_chain(vectorstore, groq_api_key: str):
    os.environ["GROQ_API_KEY"] = groq_api_key
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    def build_query(inputs: dict) -> str:
        return (
            f"Travel to {inputs['city']} for {inputs['duration']} days, "
            f"budget {inputs['budget']}, {inputs['travelers']}, "
            f"interested in {inputs['interests']}"
        )

    def retrieve_context(inputs: dict) -> str:
        query = build_query(inputs)
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        RunnablePassthrough()
        | {
            "context": retrieve_context,
            "city": lambda x: x["city"],
            "budget": lambda x: x["budget"],
            "duration": lambda x: x["duration"],
            "travelers": lambda x: x["travelers"],
            "interests": lambda x: x["interests"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chain = create_rag_chain(vectorstore, os.getenv("GROQ_API_KEY", ""))
    response = chain.invoke({
        "city": "Paris",
        "budget": "€500",
        "duration": "3",
        "travelers": "couple",
        "interests": "food, museums, romantic places",
    })
    print(response)