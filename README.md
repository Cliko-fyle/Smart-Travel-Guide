# 🗺️ Smart Travel Guide
An AI-powered Travel Recommendation System that leverages Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and semantic search to deliver personalized travel insights. Built with Hugging Face Transformers, SentenceTransformers embeddings, FAISS vector search, and OpenTripMap API.

🚀 **Live Demo**: https://cliko-fyle-smart-travel-guide.streamlit.app/

✨**Features:**

🌍 City-based Retrieval – Fetch attractions and geolocation data via OpenTripMap API\
🌐 Multilingual Support – Translate descriptions from 40+ languages into English using Hugging Face MarianMT\
🔎 Semantic Search – Encode attraction descriptions with SentenceTransformers (MiniLM, 384-dim) and retrieve results using FAISS <100ms\
🤖 RAG Pipeline with LLM – Contextual answer generation using Google Long-T5 (Hugging Face) for concise travel recommendations\
🎛️ Interactive UI – Deployed as a Streamlit web app, tested in Google Colab

🛠️ **Tech Stack:**

Frameworks/Libraries: Streamlit, Hugging Face Transformers, SentenceTransformers, FAISS, PyTorch\
Models: MarianMT (translation), Long-T5 (summarization), MiniLM (embeddings)\
APIs: OpenTripMap API (city coordinates + attraction data)\
Deployment: Google Colab + ready for Streamlit Cloud

⚙️ **Usage:**

Enter a destination city (e.g., Paris)\
The app will fetch top attractions and build a semantic knowledge base\
Enter a query (e.g., “famous museums in Paris”)\
The app retrieves relevant attractions and uses RAG + LLM to generate concise recommendations
