import streamlit as st
import os
import requests
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

api_key = st.secrets["API_KEY"]
#helper functions

def get_city_coordinates(city_name, api_key):
    url = f"https://api.opentripmap.com/0.1/en/places/geoname"
    params = {"name": city_name, "apikey": api_key}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("lat"), data.get("lon")
    return None, None


def get_top_attractions(lat, lon, api_key, radius=20000, limit=20):
    url = "https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        "radius": radius,
        "lon": lon,
        "lat": lat,
        "rate": 3,
        "format": "json",
        "limit": limit,
        "apikey": api_key
    }
    r = requests.get(url, params=params)
    return r.json() if r.status_code == 200 else []


def get_attraction_details(xid, api_key):
    url = f"https://api.opentripmap.com/0.1/en/places/xid/{xid}"
    params = {"apikey": api_key}
    r = requests.get(url, params=params)
    return r.json() if r.status_code == 200 else {}


def translate_to_english(text, tokenizer, model):
    if not text:
        return ""
    batch = tokenizer([text], return_tensors="pt", padding=True)
    generated = model.generate(**batch)
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    return translated


def build_index(descriptions, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(descriptions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return embed_model, index, embeddings


def retrieve_similar_places(query, embed_model, index, metadata, texts, top_k=5):
    query_vector = embed_model.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "name": metadata[idx],
            "description": texts[idx],
            "distance": float(distances[0][i])
        })
    return results

def call_llm(prompt):
    client = Groq(
        api_key = st.secrets["groq_api_key"]
    )
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user","content": prompt}],
        max_tokens = 300,
        )
    return response.choices[0].message.content

#load the models

@st.cache_resource
def load_translation_model():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    return tokenizer, model

# @st.cache_resource
# def load_rag_model():
#     tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")
#     rag_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
#     return rag_pipeline


#streamlit app

st.title("🗺️ Smart Travel Guide")

city = st.text_input("Enter destination city", placeholder="e.g., Paris")

if city:
    with st.spinner("Fetching attractions..."):
        lat, lon = get_city_coordinates(city, api_key)
        if lat and lon:
            attractions = get_top_attractions(lat, lon, api_key, radius=20000, limit=20)

            # load models
            trans_tokenizer, trans_model = load_translation_model()
            # rag_pipeline = load_rag_model()

            detailed_attractions = []
            for a in attractions:
                xid = a.get("xid")
                if xid:
                    details = get_attraction_details(xid, api_key)
                    name = details.get("name", "Unknown")
                    desc = details.get("wikipedia_extracts", {}).get("text", "")
                    translated_desc = translate_to_english(desc, trans_tokenizer, trans_model)
                    detailed_attractions.append({"name": name, "description": translated_desc})

            if detailed_attractions:
                texts = [d['description'] for d in detailed_attractions]
                metadata = [d['name'] for d in detailed_attractions]

                embed_model, index, embeddings = build_index(texts)

                st.success(f"Knowledge base built for {city} ✅")

                query = st.text_input("Ask a travel-related question", placeholder="e.g., famous museums in Paris")
                
                if query:
                    results = retrieve_similar_places(query, embed_model, index, metadata, texts)

                    # Build context
                    context = "\n\n".join([f"{r['name']}: {r['description']}" for r in results])
                    prompt = f"""
                    Question: {query}

                    Context:
                    {context}

                    Instruction: Summarize all relevant places in a concise list with a brief description of each place.
                    """

                    with st.spinner("Generating answer..."):
                        answer = call_llm(prompt)

                    st.subheader("🔎 Suggested Places")
                    st.write(answer[0]['generated_text'])
        else:

            st.error("City not found. Try another.")







