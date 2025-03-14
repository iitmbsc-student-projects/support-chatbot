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

def create_vector_store_from_json_using_subject(json_file_path, vector_store_path):
    """
    Create a FAISS vector store using only 'subject' field for embeddings.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_documents = []
    total_characters = 0

    for entry in data:
        subject = entry.get('subject', '').strip()  # Use only subject for similarity search
        description = entry.get('description', '').strip()  # Store description as metadata only

        document = Document(
            page_content=subject,  # ✅ Store only subject for embeddings
            metadata={"subject": subject, "description": description}  # ✅ Keep description in metadata
        )

        all_documents.append(document)
        total_characters += len(subject)  # Count only subject characters

    # Text splitting is optional for short subjects, but keeping it in case of long text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128000,  
        chunk_overlap=200,  
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load existing vector store if available, otherwise create a new one
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
        document_ids = vector_store.add_documents(documents=split_documents)
    else:
        vector_store = FAISS.from_documents(split_documents, embeddings)
        document_ids = list(range(len(split_documents)))  # Assign dummy IDs

    vector_store.save_local(vector_store_path)

    return vector_store, total_characters, len(data), split_documents, len(document_ids)

def load_vector_store(vector_store_path):
    """
      This function is used to load a vector store that has been saved to disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # This is the model that was used to create the vector store
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)


def get_similar_queries(query, vector_store, k=5):
    """
    Find similar queries from the vector store.
    """
    print("\n🔍 Searching for queries similar to:", query)
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Get query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Fetch similar documents
    similar_documents = vector_store.similarity_search_by_vector(query_embedding, k=k)
    
    # Extracting queries from the results
    # print(f"%%%%%%%\n\n{similar_documents[0].page_content}\n\n%%%%%%%%%\n\n")
    similar_queries = [doc.page_content.strip().replace("\r", "").replace("\n", " ").split("Description")[0] for doc in similar_documents]
    
    # Print results in a readable format
    print("\n📌 **Top {} Similar Queries:**".format(k))
    for i, q in enumerate(similar_queries, 1):
        print(f"{i}. {q}")
    
    return similar_queries

# if __name__ == '__main__':
#     vector_store = load_vector_store("vector_store_only_subject")
#     similar_queries = get_similar_queries("Quiz 1", vector_store, k=5)
    # print(similar_queries)