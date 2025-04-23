import os
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
import torch
from pprint import pprint
from IPython.display import Markdown as md
from groq import Groq
import yaml

# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)
model = None

# groq_api_key = config["GROQ_API_KEY"]
# os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

groq_api_key = os.environ["GROQ_API_KEY"]

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

system_prompt = """
You are an AI assistant trained to answer questions based on the content of the PDF document provided, which is a student handbook.
The document is highly detailed and contains knowledge about the entire IIT Madras BS Degree.
When asked a question, you should refer to the document and provide the most accurate and relevant answer based on the information in the document.
Answer only based on the content from the document and do not make up any information.
"""

client = Groq(api_key=groq_api_key)

def query_model_with_rag(query, vector_store):
    # print(f" Recieved query is {query}")
    relevant_chunks = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    # print(f" Recieved CONTEXT is {context}")
    prompt = system_prompt + "\n\nContext:\n" + context + "\n\nQuestion: " + query + "\nAnswer:"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        stream=False
    )
    return chat_completion.choices[0].message.content.strip()