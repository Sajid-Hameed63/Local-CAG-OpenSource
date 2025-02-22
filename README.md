# Chat with Multiple PDFs 📚

This is a **Streamlit-based application** that allows users to upload multiple PDF files and ask questions about their content using **Google Gemini AI** and **FAISS for vector storage**.

## Features
- 📄 **Upload multiple PDFs** and process their text.
- 🧠 **Generate embeddings** using Google Gemini AI.
- 🔍 **Store and retrieve document embeddings** with FAISS.
- 🤖 **Answer user queries** based on document context.
- 💾 **Persistent document index** to avoid reprocessing.
- 🗨️ **Chat history storage** for better interaction.

## Deployment
🚀 This application is **deployed on Hugging Face Spaces** for easy access and usage. You can try it out [here](https://huggingface.co/spaces/sajidhameed63/Chat-with-Your-PDFs).

## Docker Support
You can also run this application using **Docker**:

### Build the Docker Image
```sh
sudo docker build -t rag-chat-multiple-pdfs:v1.0 .
```

### Run the Container
```sh
docker run --env-file .env -p 8500:8500 rag-chat-multiple-pdfs:v1.0
```

### Pull from Docker Hub
Alternatively, you can pull the pre-built image from Docker Hub:
```sh
docker pull sajidhameed63/rag-chat-multiple-pdfs:v1.0
```

## Requirements
Ensure you have the following installed:

- Python 3.8+
- Streamlit
- PyPDF2
- LangChain
- FAISS
- Google Gemini API Key (Required for embeddings and chat model)
- dotenv (for managing API keys)

## Installation
```sh
# Clone the repository
git clone https://github.com/Sajid-Hameed63/Chat-Multiple-PDFs.git
cd Chat-Multiple-PDFs

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup
1. **Get a Google Gemini API Key**
   - Visit [Google AI Studio](https://aistudio.google.com/) and generate an API key.
   - Create a `.env` file in the root directory and add:
     ```env
     GOOGLE_GEMINI_API_KEY=your_api_key_here
     ```

2. **Run the application**
   ```sh
   streamlit run app.py
   ```

## Usage
1. Upload PDF files in the sidebar.
2. Click **Process PDFs** to extract text and generate embeddings.
3. Ask questions about your documents using the chat input.
4. View conversation history and clear chat if needed.

## Troubleshooting
- **PDF processing failed?** Ensure the files contain selectable text.
- **Embedding errors?** Check if your Google API key is valid.
- **Vector store not found?** Make sure PDFs are processed first.

## Future Improvements
- Multi-user session handling.
- Support for scanned PDFs (OCR-based extraction).
- Integration with other LLMs (e.g., OpenAI GPT).

## License
MIT License

---
**Developed by Sajid Hameed** 🚀

