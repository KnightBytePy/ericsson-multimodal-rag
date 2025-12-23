# 📶 Ericsson Mobility Report: Multi-Modal RAG Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link-here)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-magenta)

## 📋 Project Overview
This project is a **Multi-Modal Retrieval-Augmented Generation (RAG)** application designed to answer technical questions based on the **Ericsson Mobility Report**. 

Unlike standard text-only RAG systems, this prototype ingests and retrieves information from both **text and visual data** (charts, graphs, and figures) contained within the PDF, allowing for a complete analysis of the source material.

## 🎯 Motivation
This prototype was built to demonstrate practical solutions to technical challenges discussed during my interview process:
1.  **Multi-Modal Ingestion (Re: Mohammad's Query):** How to handle complex PDFs containing both text and dense statistical figures.
2.  **Scalability & Parallelism (Re: Khao's Query):** How to structure data pipelines to allow for parallel processing and efficient scaling.

## 🏗️ Technical Architecture

### 1. Ingestion Pipeline & OCR Strategy
* **Tool:** `PyMuPDF` (Fitz)
* **Decision:** I evaluated **Tesseract** (OCR) vs. **PyMuPDF**. 
    * *Trade-off:* While Tesseract is powerful, it is computationally expensive and slow for large reports.
    * *Solution:* I selected `PyMuPDF` for its low-latency extraction of text and images, ensuring the app remains responsive even in a local environment.

### 2. Scalability & Chunking
* **Method:** `RecursiveCharacterTextSplitter`
* **Why:** To address scalability, I avoided simple token splitting. By using recursive splitting with defined overlap, I created semantically independent chunks. This structure is the prerequisite for **parallel embedding pipelines**, allowing the system to scale horizontally in a production environment.

### 3. The Brain (LLM & Vector Store)
* **Model:** Google **Gemini 2.0 Flash** (via `langchain-google-genai`).
    * *Optimization:* Configured with low temperature (0.0 - 0.1) for factual accuracy.
* **Vector Database:** `ChromaDB` (Persistent local storage).
* **Embeddings:** `text-embedding-004`.

### 4. Guardrails
* The system includes strict prompt engineering to prevent **hallucinations**.
* It filters out irrelevant queries (e.g., "How to make a cake") to ensure the agent remains a focused technical assistant.

---

## 🚀 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ericsson-multimodal-rag.git](https://github.com/YOUR_USERNAME/ericsson-multimodal-rag.git)
    cd ericsson-multimodal-rag
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys**
    Create a `.streamlit/secrets.toml` file:
    ```toml
    GOOGLE_API_KEY = "your_google_api_key_here"
    ```

4.  **Run the App**
    ```bash
    streamlit run src/app.py
    ```

---

## 🔮 Future Roadmap
* **Graph Neural Networks (GNNs):** Exploring the application of GNNs for network optimization and topology analysis, moving beyond retrieval into predictive modeling.
* **Containerization:** Dockerizing the application for cloud deployment (Kubernetes/OpenShift).

## 👤 Author
**Fares Jony** [LinkedIn](https://www.linkedin.com/in/faresjony/)
