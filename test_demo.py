import os
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModel
import torch
from pprint import pprint
from IPython.display import Markdown as md
import re
from groq import Groq
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
model = None

groq_api_key = config["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

def pdf_to_text(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

pdf_path = "./documents/IITM BS Degree Programme - Student Handbook.pdf"
print(f"pdf_path: {pdf_path}")
pdf_text = pdf_to_text(pdf_path)

def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    cleaned_sentences = []
    for sent in sentences:
      sent = sent.replace("\n", " ")
      sent = re.sub(r'\s+', ' ', sent)
      cleaned_sentences.append(sent)
    return cleaned_sentences

def chunk_text_with_some_overlap(text, chunk_size=5, overlap=2):
    initial_chunks, final_chunks = [], []
    clean_sentences = split_text_into_sentences(text)
    concatenated_text = " ".join(clean_sentences)
    spaced_list = concatenated_text.split()

    step = chunk_size - overlap
    for i in range(0, len(spaced_list), step):
      initial_chunks.append(spaced_list[i:i + step])

    for i in range(0, len(initial_chunks)-1):
      temp = list(initial_chunks[i])
      temp.extend(initial_chunks[i+1][:overlap])
      final_chunks.append(' '.join(temp))

    return final_chunks

chunks = chunk_text_with_some_overlap(pdf_text, chunk_size=300, overlap=60)
# for chunk in chunks[7:10]:
#   print(chunk, end="\n********\n") #comment this line

# len(chunks)

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

"""# Creating the Vector Store from the text in terms of chunks taking the help of HF embeddings."""

vector_store = FAISS.from_texts(chunks, hf_embeddings)

system_prompt = """
You are an AI assistant trained to answer questions based on the content of the PDF document provided, which is a student handbook.
The document is highly detailed and contains knowledge about the entire IIT Madras BS Degree.
When asked a question, you should refer to the document and provide the most accurate and relevant answer based on the information in the document.
Answer only based on the content from the document and do not make up any information.
"""

client = Groq(api_key=groq_api_key)

def query_model_with_rag(query):
    relevant_chunks = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    prompt = system_prompt + "\n\nContext:\n" + context + "\n\nQuestion: " + query + "\nAnswer:"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        stream=False
    )
    return chat_completion.choices[0].message.content.strip()

# query = "What is the total amount of credits required to pass the BS Degree?"
# answer = query_model_with_rag(query)
# print(answer)

# query = "What is the fee of level 3 courses?"
# answer = query_model_with_rag(query)
# print(answer)

# query = "How can I get minor in Economics and Finance"
# answer = query_model_with_rag(query)
# md(answer)

# query = "What is the validity of the qualifier score? By when can I used it?"
# answer = query_model_with_rag(query)
# md(answer)

# query = "What is the learner life cycle?"
# answer = query_model_with_rag(query)
# md(answer)

# query = "How will the assignments have to be submitted and how will the be graded?"
# answer = query_model_with_rag(query)
# md(answer)

query = "Explain to me the entire passing criteria for each course?"
answer = query_model_with_rag(query)
print(answer)