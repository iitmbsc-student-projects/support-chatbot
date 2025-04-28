import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import re, requests
from bs4 import BeautifulSoup, NavigableString, Tag

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
            page_content=subject,  # Store only subject for embeddings
            metadata={"subject": subject, "description": description}  # Keep description in metadata
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
    print("\nSearching for queries similar to:", query)
    
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
    print(f"\n**Top {k} Similar Queries:**")
    for i, q in enumerate(similar_queries, 1):
        print(f"{i}. {q}")
    
    return similar_queries

def create_pdf_chunks(pdf_path, chunk_size=200, overlap=50): # This function will be used to chunk the Student Handbook
    # Step 1: Convert PDF to text
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() # whole text of the pdf as single string

    # Step 2: Clean up the text
    """Splits text into sentences using regex."""
    sentences = re.split(r'(?<=[.?!])\s+', text)  # Split at sentence boundaries
    cleaned_sentences = []
    for sent in sentences:
      sent = sent.replace("\n", " ")  # Replace newlines with spaces
      sent = re.sub(r'\s+', ' ', sent)  # Replace multiple spaces with a single space
      cleaned_sentences.append(sent)

    # Step 3: Now we have the whole text of the PDF, create overlapping chunks
    """Splits a long text into chunks of max `chunk_size` words with `overlap`."""
    initial_chunks, final_chunks = [], [] # initial_chunks are non-overlapping; final_chunks are overlapping
    clean_sentences = list(cleaned_sentences)
    concatenated_text = " ".join(clean_sentences) # Concatenate separated sentences into a single string
    spaced_list = concatenated_text.split() # Split concatenated text into a list of words

    step = chunk_size - overlap
    for i in range(0, len(spaced_list), step): # create non-overlapping chunks
      # print(f"i={i}")
      initial_chunks.append(spaced_list[i:i + step])

    for i in range(0, len(initial_chunks)-1): # create overlapping chunks
      temp = list(initial_chunks[i]) # Make a copy
      # print(f"TEMP = {temp}")
      temp.extend(initial_chunks[i+1][:overlap])
      final_chunks.append(' '.join(temp))

    return final_chunks

def create_grading_doc_chunks(grading_doc_url): # We have used URL because it's much easier to chunk the grading doc from URL than from a PDF
    # Step 1: Get the text from the URL
    try:
        response = requests.get(grading_doc_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Error fetching the URL: {e}")
        return []

    segments = []
    current_segment = []

    def traverse(node):
        nonlocal segments, current_segment

        if isinstance(node, NavigableString):
            text = node.strip()
            if text:
                current_segment.append(text)
        elif isinstance(node, Tag):
            # Skip tags that typically don't contribute visible content
            if node.name in ["script", "style"]:
                return

            if node.name == "table":
                if current_segment:
                    segments.append(" ".join(current_segment).replace("\xa0",""))
                    current_segment.clear()
                # Skip the children of the table entirely
                return
            else:
                for child in node.children:
                    traverse(child)

    # If you know the main content is in a particular container, adjust this
    start_node = soup.body if soup.body else soup
    traverse(start_node)

    if current_segment:
        segments.append(" ".join(current_segment).replace("\xa0",""))
    keep_indices = [i for i in range(len(segments)) if i not in (0, 1, 3)] # delete 0, 1, 3 indices
    final_chunks = [segments[i] for i in keep_indices]
    # D = {}
    # for i,chunk in enumerate(final_chunks):
    #     D[f"i"] = chunk
    # with open("GD_chunks.json","w") as f:
    #     json.dump(f, D, indent=4)

    return final_chunks

from bs4 import BeautifulSoup
import requests
import re

def extract_chunks_by_header_without_tables(url):
    # Step 1: Fetch and clean HTML
    response = requests.get(url)
    html = response.text
    cleaned_html = re.sub(r'<table.*?>.*?</table>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Step 2: Parse with BeautifulSoup
    soup = BeautifulSoup(cleaned_html, 'html.parser')

    # Step 3: Extract chunks
    chunks = {}
    headers = soup.find_all(['h1', 'h2'])

    for i, header in enumerate(headers):
        key = header.get_text(strip=True)
        content = []

        # Traverse next siblings until the next header
        for sibling in header.find_next_siblings():
            if sibling.name and sibling.name.lower() in ['h1', 'h2']:
                break
            text = sibling.get_text(strip=True, separator=' ')
            if text:
                content.append(text)

        chunks[key] = ' '.join(content).replace("&nbsp:","").replace("\u00a0","").replace("\u00b7","").replace("\u2014","").replace("\u00a0","").replace("\u2013","")
    
    keys_to_remove = ["ASSIGNMENT DEADLINES:","Bonus Marks","Suggested pathway to register and study Foundation level courses:","Diploma Level courses","Suggested pathway to register and study Diploma level courses:","Diploma level courses","Degree Level courses","Annexure I"]
    for key in chunks.keys():
        if (key.strip()=="" or chunks[key].strip()=="") and (key not in keys_to_remove):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        chunks.pop(key)

    final_chunks = [f"Grading policy for {key}:\t {value}\n" for key, value in chunks.items()]

    return final_chunks

def create_vector_store_from_docs(pdf_path, grading_doc_url):
    pdf_chunks = create_pdf_chunks(pdf_path)
    # grading_doc_chunks = create_grading_doc_chunks(grading_doc_url)
    new_grading_doc_chunks = extract_chunks_by_header_without_tables(grading_doc_url)
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(pdf_chunks+new_grading_doc_chunks, hf_embeddings)
    return vector_store


if __name__ == "__main__":
    pdf_path = "documents/Student_Handbook_latest.pdf"
    grading_doc_url = "https://docs.google.com/document/d/e/2PACX-1vRBH1NuM3ML6MH5wfL2xPiPsiXV0waKlUUEj6C7LrHrARNUsAEA1sT2r7IHcFKi8hvQ45gSrREnFiTT/pub?urp=gmail_link"
    latest_vector_store = create_vector_store_from_docs(pdf_path, grading_doc_url)
    vector_store_path = "LATEST_VECTOR_STORE"
    latest_vector_store.save_local(vector_store_path)

    