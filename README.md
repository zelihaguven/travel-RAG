# ✈️ AI Travel Assistant — RAG-Powered Trip Planner

A personalised travel planning chatbot built with **Retrieval-Augmented Generation (RAG)**, powered by real TripAdvisor reviews and destination data. Users input their destination, budget, duration, and interests — and receive a detailed, AI-generated travel plan with real restaurant names, hotel suggestions, activities, and a day-by-day itinerary.

---

## 🧠 How It Works

```
Real TripAdvisor Reviews + Destination Data
                ↓
        Data Cleaning & Preprocessing
                ↓
     Text Chunking (LangChain Text Splitter)
                ↓
   Embedding Generation (all-MiniLM-L6-v2)
                ↓
        ChromaDB Vector Storage
                ↓
          User Inputs Trip Details
                ↓
     Semantic Search → Top 12 Relevant Reviews
                ↓
   Groq LLM (LLaMA 3.3 70B) Generates Plan
                ↓
        Structured Travel Plan Output
```

---

## 🚀 Features

- 🏨 **Hotel Recommendations** — matched to your budget
- 🍽️ **Restaurant Picks** — based on real traveller reviews with specific place names
- 🎯 **Activities & Attractions** — aligned with your interests
- 📅 **Day-by-Day Itinerary** — hour-by-hour plan for each day
- 💰 **Budget Breakdown** — estimated spend per category
- 💡 **Money-Saving Tips** — practical advice per destination

---

## 🗺️ Supported Cities

| City | Reviews Available |
|------|:-----------------:|
| 🗼 Paris | ✅ |
| 🎡 London | ✅ |
| 🥘 Barcelona | ✅ |
| 🏛️ Madrid | ✅ |
| 🗽 New York | ✅ |
| 🕌 New Delhi | ✅ |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| RAG Framework | LangChain 0.3.27 |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| UI | Streamlit |
| Data Source | TripAdvisor Reviews via Kaggle |

---

## 📁 Project Structure

```
travel_rag/
├── chroma_db/          # Vector database (auto-generated)
├── data/               # Raw datasets (auto-generated)
├── ingest.py           # Data download, cleaning & ChromaDB ingestion
├── rag.py              # RAG pipeline & LangChain chain
├── app.py              # Streamlit web interface
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/travel-rag-assistant.git
cd travel-rag-assistant
```

### 2. Create a virtual environment

```bash
python -m venv rag_travel
source rag_travel/bin/activate      # Mac/Linux
rag_travel\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Ingest data into ChromaDB

```bash
python ingest.py
```

This will:
- Download 5 datasets from Kaggle (~600 MB total)
- Clean and merge ~2.7 million reviews
- Generate embeddings and store 5,000 chunks in ChromaDB

> ⏱️ This step takes around 5–10 minutes depending on your internet speed.

### 6. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 📊 Datasets Used

| Dataset | Kaggle Author | Size |
|---------|--------------|------|
| TripAdvisor Hotel Reviews | andrewmvd | ~20K reviews |
| TripAdvisor Reviews 2023 (New Delhi) | arnabchaki | ~147K reviews |
| TripAdvisor City Reviews (6 cities) | inigolopezrioboo | ~2.6M reviews |
| Travel Recommendation Dataset | amanmehra23 | Destination info |
| Traveler Trip Dataset | rkiattisak | Trip cost data |

---

## 🎓 Academic Context

This project was developed as part of the **"Datenanalyse in Unternehmen"** (Data Analysis in Companies) lecture at **Hochschule Heilbronn**, supervised by **Prof. Dr. Christine Reck**.

It falls under **Option 7 — Large Language Models & Chatbots**, specifically focusing on:

- Semantic Search using LLM-generated embeddings
- Retrieval Augmented Generation (RAG)
- Vector databases for efficient similarity search

---

## 👥 Authors

- **Ilgın Güven** — Hochschule Heilbronn, Software Engineering

---

## 📄 License

MIT License — feel free to use and modify!
