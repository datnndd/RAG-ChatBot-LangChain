import os
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# =========================
# CONFIG
# =========================
load_dotenv()
VECTOR_DB_DIR = "vector_store_chroma"

# =========================
# INIT COMPONENTS
# =========================
def initialize_components():
    """Khởi tạo LLM, embeddings và vector store"""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    #Thay mô hình LLM nếu bị hạn chế
    #gemini-2.5-flash-lite
    #gemini-3-flash
    #gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    if not os.path.exists(VECTOR_DB_DIR):
        raise RuntimeError("Chưa có Vector DB. Hãy chạy build_vector_db.py trước!")

    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return llm, retriever


def format_docs(docs):
    """Format danh sách documents thành chuỗi context"""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        if meta.get("doc_type") == "product":
            formatted.append(
                f"Sản phẩm: {meta.get('product_name')}\n"
                f"- Giá: {meta.get('price')}đ\n"
                f"- Màu: {meta.get('color')}\n"
                f"- Size: {meta.get('size')}\n"
                f"- Tồn kho: {meta.get('stock')}\n"
                f"- Đánh giá: {meta.get('rating')}\n"
                f"- Mô tả: {doc.page_content}"
            )
        else:
            formatted.append(f"Tài liệu ({meta.get('source', 'unknown')}): {doc.page_content}")
    return "\n\n---\n".join(formatted)


def format_sources(docs):
    """Format nguồn tham khảo từ documents"""
    sources = set()
    for doc in docs:
        meta = doc.metadata
        if meta.get("doc_type") == "product":
            sources.add(
                f"🛍️ {meta.get('product_name')} | "
                f"{meta.get('price'):,}đ | "
                f"{meta.get('color')} | "
                f"Size {meta.get('size')} | "
                f"Tồn: {meta.get('stock')}"
            )
        else:
            sources.add(f"📄 {meta.get('source', 'unknown')}")
    return sources


# =========================
# INIT LLM & RETRIEVER
# =========================
llm, retriever = initialize_components()

# Lưu lịch sử chat
chat_history = []

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI tư vấn thời trang cho cửa hàng Uqilo.
Nhiệm vụ của bạn:
- Tư vấn sản phẩm dựa trên thông tin được cung cấp
- Trả lời câu hỏi về thông tin công ty
- Gợi ý sản phẩm phù hợp với nhu cầu khách hàng

Thông tin tham khảo:
{context}

Lưu ý:
- Chỉ trả lời dựa trên thông tin được cung cấp
- Nếu không có thông tin, hãy nói rõ là không biết
- Trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


# =========================
# CHAT HANDLER
# =========================
def chat_handler(message, history):
    """Xử lý tin nhắn từ Gradio"""
    global chat_history
    
    if not message.strip():
        return ""
    
    try:
        # Lấy documents
        docs = retriever.invoke(message)
        context = format_docs(docs)
        
        # Tạo chain và gọi
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": message,
            "context": context,
            "chat_history": chat_history
        })
        
        # Cập nhật lịch sử
        chat_history.append(HumanMessage(content=message))
        chat_history.append(AIMessage(content=response))
        
        # Giữ tối đa 10 tin nhắn
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
        
        # Thêm nguồn tham khảo
        sources = format_sources(docs)
        if sources:
            response += "\n\n---\n**🔍 Nguồn tham khảo:**\n"
            response += "\n".join(f"- {s}" for s in sources)
        
        return response
        
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"


# =========================
# GRADIO UI (Gradio 6.0+)
# =========================
def main():
    demo = gr.ChatInterface(
        fn=chat_handler,
        title="🛍️ Uqilo Fashion Chatbot",
        description="Trợ lý AI tư vấn thời trang thông minh",
        examples=[
            "Áo màu đỏ dưới 300k",
            "Quần size L còn hàng",
            "Sản phẩm đánh giá trên 4.5"
        ]
    )
    
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()