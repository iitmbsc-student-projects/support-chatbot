from flask import Flask, render_template, request
from vector_store import create_vector_store_from_json, create_vector_store_from_json_using_subject, load_vector_store, get_similar_queries

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search_similar_queries', methods=['POST'])
def search_similar_queries():
    query = request.form['query']
    similar_queries = get_similar_queries(query, vector_store, k=5)
    return render_template('index.html', similar_queries=similar_queries)

if __name__ == '__main__':
    json_file_path = "documents/tickets_3k.json"  
    vector_store_path = "vector_store_only_subject"
    vector_store = load_vector_store(vector_store_path) 

    # vector_store, total_characters, total_entries, split_documents, total_added_docs = vectorStore.create_vector_store_from_json_using_subject(
    #     json_file_path, vector_store_path
    # )
    # print(f"Vector store created with {total_entries} entries and {total_added_docs} documents added.")
    app.run(debug=True)