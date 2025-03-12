import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store_from_json(json_file_path, vector_store_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    all_documents = []
    total_characters = 0

    for entry in data:
        subject = entry.get('subject', '')
        description = entry.get('description', '')

        doc_content = f"Subject: {subject}\nDescription: {description}"

        document = Document(
            page_content=doc_content,
            metadata={"subject": subject, "description": description}
        )

        all_documents.append(document)
        total_characters += len(doc_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128000,  
        chunk_overlap=200,  
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(split_documents, embeddings)

    document_ids = vector_store.add_documents(documents=split_documents)
    vector_store.save_local(vector_store_path)

    return vector_store, total_characters, len(data), split_documents, len(document_ids)