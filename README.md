# 🛍️ Uqilo Fashion Chatbot

A Vietnamese AI-powered fashion assistant chatbot built with **LangChain**, **Google Gemini**, and **Gradio**. The chatbot uses RAG (Retrieval-Augmented Generation) to provide intelligent product recommendations and answer questions about the Uqilo clothing store.

## ✨ Features

- 🤖 **AI-Powered Conversations** - Uses Google Gemini for natural language understanding
- 🔍 **Smart Product Search** - Semantic search through product catalog using ChromaDB
- 💬 **Chat History** - Maintains conversation context for better responses
- 📄 **Multi-Source Knowledge** - Supports both CSV product data and DOCX documents
- 🌐 **Web Interface** - User-friendly Gradio chat interface

## 📁 Project Structure

```
AIChat/
├── chatbot.py           # Main chatbot application with Gradio UI
├── build_vector_db.py   # Script to build ChromaDB vector store
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── knowledge-base/      # Source data for the chatbot
│   ├── company/         # Company information (DOCX files)
│   └── product/         # Product catalog (CSV files)
└── vector_store_chroma/ # Generated vector database
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini AI)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AIChat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Usage

1. **Build the Vector Database**
   
   First, prepare your knowledge base by placing:
   - Product CSV files in `knowledge-base/product/`
   - Company documents (DOCX) in `knowledge-base/company/`

   Then run:
   ```bash
   python build_vector_db.py
   ```

2. **Start the Chatbot**
   ```bash
   python chatbot.py
   ```

   The web interface will open automatically at `http://127.0.0.1:7860`

## 📊 Data Format

### Product CSV Structure

| Column      | Description           |
|-------------|-----------------------|
| MaSanPham   | Product ID            |
| TenSanPham  | Product Name          |
| DanhMuc     | Category              |
| MauSac      | Color                 |
| KichThuoc   | Size (S, M, L, XL...) |
| GiaTien     | Price (VND)           |
| TonKho      | Stock quantity        |
| DanhGia     | Rating (1-5)          |
| MoTa        | Description           |

## 💡 Example Queries

- "Áo màu đỏ dưới 300k" (Red shirts under 300k)
- "Quần size L còn hàng" (Size L pants in stock)
- "Sản phẩm đánh giá trên 4.5" (Products rated above 4.5)

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| [LangChain](https://python.langchain.com/) | LLM orchestration framework |
| [Google Gemini](https://ai.google.dev/) | Large Language Model |
| [ChromaDB](https://www.trychroma.com/) | Vector database |
| [Gradio](https://gradio.app/) | Web UI framework |

## 📝 License

This project is for educational purposes.

---

*Built with ❤️ for Uqilo Fashion Store*
