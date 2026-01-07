import os
import shutil
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

# =========================
# CONFIG
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

KNOWLEDGE_BASE_DIR = "knowledge-base"          
VECTOR_DB_DIR = "vector_store_chroma"

# =========================
# CSV LOADER
# =========================
def load_products_csv(file_path):
    documents = []

    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        try:
            ma_sp = str(row["MaSanPham"]).strip()
            ten_sp = str(row["TenSanPham"]).strip()
            danh_muc = str(row["DanhMuc"]).strip().lower()
            mau_sac = str(row["MauSac"]).strip().lower()
            kich_thuoc = str(row["KichThuoc"]).strip().upper()
            mo_ta = str(row["MoTa"]).strip()

            # Giá tiền
            raw_price = str(row["GiaTien"])
            price_int = int("".join(filter(str.isdigit, raw_price)))

            ton_kho = int(row["TonKho"])
            danh_gia = float(row["DanhGia"])

            content = (
                f"Sản phẩm: {ten_sp}\n"
                f"Mã sản phẩm: {ma_sp}\n"
                f"Danh mục: {danh_muc}\n"
                f"Màu sắc: {mau_sac}\n"
                f"Kích thước: {kich_thuoc}\n"
                f"Giá bán: {price_int} VNĐ\n"
                f"Tồn kho: {ton_kho}\n"
                f"Đánh giá: {danh_gia}/5\n"
                f"Mô tả: {mo_ta}"
            )

            metadata = {
                "doc_type": "product",
                "source": os.path.basename(file_path),

                "product_id": ma_sp,
                "product_name": ten_sp,
                "category": danh_muc,
                "color": mau_sac,
                "size": kich_thuoc,
                "price": price_int,
                "stock": ton_kho,
                "rating": danh_gia
            }

            documents.append(
                Document(page_content=content, metadata=metadata)
            )

        except Exception as e:
            print(f"Bỏ qua dòng lỗi trong {file_path}: {e}")

    print(f"Load CSV: {os.path.basename(file_path)} ({len(documents)} sản phẩm)")
    return documents

# =========================
# LOAD ALL DOCUMENTS
# =========================
def load_documents(root_folder):
    product_docs = []
    text_docs = []

    for root, _, files in os.walk(root_folder):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".csv"):
                product_docs.extend(load_products_csv(path))

            elif file.endswith(".docx"):
                loader = Docx2txtLoader(path)
                docs = loader.load()
                for d in docs:
                    d.metadata["doc_type"] = "document"
                    d.metadata["source"] = file
                text_docs.extend(docs)

    # Chunk ONLY text documents
    if text_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        text_docs = splitter.split_documents(text_docs)

    return product_docs, text_docs

# =========================
# BUILD VECTOR DB
# =========================
def build_vector_db():
    if not GOOGLE_API_KEY:
        raise RuntimeError("Chưa có GOOGLE_API_KEY")

    product_docs, text_docs = load_documents(KNOWLEDGE_BASE_DIR)
    all_docs = product_docs + text_docs

    if not all_docs:
        print("Không có dữ liệu để build vector DB")
        return

    print("========== SUMMARY ==========")
    print(f"Products: {len(product_docs)}")
    print(f"Documents: {len(text_docs)}")
    print(f"TOTAL: {len(all_docs)}")
    print("=============================")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    print(f"✅ Hoàn tất! Vector DB có {len(all_docs)} documents")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    build_vector_db()
