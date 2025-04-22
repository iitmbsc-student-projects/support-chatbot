from flask import Flask, render_template, request, jsonify
import os

from vector_store import load_vector_store, create_vector_store_from_docs
from ask_llm import query_model_with_rag

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/search_similar_queries', methods=['POST'])
# def search_similar_queries():
#     query = request.form['query']
#     similar_queries = get_similar_queries(query, vector_store, k=5)
#     return render_template('index.html', similar_queries=similar_queries)

@app.route("/get_answer_from_llm", methods=["POST"])
def get_answer_from_llm():
    query = request.form["query"]
    answer = query_model_with_rag(query, vector_store)
    return jsonify({"response": answer})


if __name__ == '__main__':
    vector_store_path = "Vector_store"

    if len(os.listdir(vector_store_path))==0:
        # Step 1: Create a vector store
        pdf_path = "documents/Student_Handbook_latest.pdf"
        grading_doc_url = "https://docs.google.com/document/d/e/2PACX-1vRBH1NuM3ML6MH5wfL2xPiPsiXV0waKlUUEj6C7LrHrARNUsAEA1sT2r7IHcFKi8hvQ45gSrREnFiTT/pub?urp=gmail_link"
        vector_store = create_vector_store_from_docs(pdf_path, grading_doc_url)
        vector_store.save_local(vector_store_path)
    else:
        vector_store = load_vector_store(vector_store_path)

    app.run(debug=True)