import os
import streamlit as st
from dotenv import load_dotenv
from rag import load_vectorstore, create_rag_chain

load_dotenv()

st.set_page_config(
    page_title="Travel Assistant",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AI Travel Assistant")
st.subheader("Personalised travel plans powered by real reviews")

@st.cache_resource(show_spinner="Loading travel knowledge base...")
def get_vectorstore():
    return load_vectorstore()

@st.cache_resource(show_spinner="Initialising AI chain...")
def get_chain(_vectorstore, api_key: str):
    return create_rag_chain(_vectorstore, api_key)

with st.sidebar:
    st.header("Trip Details")
    groq_key = os.getenv("GROQ_API_KEY", "")
    city = st.selectbox(
        "Destination City",
        ["Paris", "London", "Barcelona", "Madrid", "New York", "New Delhi"],
    )
    budget = st.slider("Budget (€)", min_value=100, max_value=5000, value=500, step=100)
    duration = st.slider("Duration (days)", min_value=1, max_value=14, value=3)
    travelers = st.selectbox(
        "Traveling as",
        ["Solo", "Couple", "Family with kids", "Group of friends"],
    )
    interests = st.multiselect(
        "Interests",
        ["Food & Restaurants", "Museums & Culture", "Shopping",
         "Nature & Parks", "Nightlife", "History", "Art", "Architecture"],
        default=["Food & Restaurants", "Museums & Culture"],
    )
    generate = st.button("Generate Travel Plan", use_container_width=True, type="primary")

if generate:
    if not groq_key:
        st.error("Please add your GROQ_API_KEY to the .env file.")
        st.stop()
    if not interests:
        st.warning("Please select at least one interest.")
        st.stop()
    try:
        vectorstore = get_vectorstore()
        rag_chain = get_chain(vectorstore, groq_key)
        with st.spinner("Creating your personalised travel plan..."):
            response = rag_chain.invoke({
                "city": city,
                "budget": f"€{budget}",
                "duration": str(duration),
                "travelers": travelers,
                "interests": ", ".join(interests),
            })
        st.success("Your travel plan is ready!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Destination", city)
        col2.metric("Budget", f"€{budget}")
        col3.metric("Duration", f"{duration} days")
        col4.metric("Travelers", travelers)
        st.divider()
        st.markdown(response)
    except RuntimeError as e:
        st.error(str(e))
        st.info("Run `python ingest.py` in the terminal to populate the knowledge base.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Fill in your trip details on the left and click **Generate Travel Plan**.")
    st.markdown("""
    ### What you get
    - **Hotel recommendations** matched to your budget
    - **Restaurant picks** based on real traveller reviews
    - **Top attractions** aligned with your interests
    - **Day-by-day itinerary** with approximate timings
    - **Budget breakdown** and money-saving tips
    """)