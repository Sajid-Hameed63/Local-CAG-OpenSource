# Cache Augmented Generation (CAG)

Cache Augmented Generation (CAG) optimizes LLM-based applications by reducing redundant API requests through caching, leading to faster response times and improved efficiency.

## Features

- **Efficient Caching:** Reduces API calls to LLMs by storing responses and reusing them for similar queries.
- **Multiple Embedding Options:** Supports both free and local embeddings.
- **Database Storage:** Uses SQLite3 and ChromaDB for caching and retrieval.

## Implementations

### 1. **Streamlit App with Ollama DeepSeek & Ollama Embeddings (100% free & local)**
   - **LLM:** Ollama DeepSeek
   - **Embeddings:** Ollama embeddings 
   - **Databases:** SQLite3 & ChromaDB

### 2. **Streamlit App with Ollama DeepSeek & Google Embeddings**
   - **LLM:** Ollama DeepSeek
   - **Embeddings:** Google embeddings (via free Gemini API keys)
   - **Databases:** SQLite3 & ChromaDB

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sajid-Hameed63/Local-CAG-OpenSource
   cd Local-CAG-OpenSource
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_cag_ollama.py 
   ```
   Or 
   ```bash
   streamlit run streamlit_cag_gemini.py # create .env and add gemini API key as GOOGLE_GEMINI_API_KEY
   ```

## Usage

- Input your query in the UI.
- The system checks if a cached response exists.
- If cached, it retrieves the response instantly.
- If not cached, it fetches a new response, stores it, and returns it.

## Contributing

Feel free to contribute by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

