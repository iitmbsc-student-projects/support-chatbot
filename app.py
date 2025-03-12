from flask import Flask, render_template
import vector_store 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    json_file_path = "documents/tickets_3k.json"  
    vector_store_path = "vector_store"  

    # vector_store, total_characters, total_entries, split_documents, total_added_docs = vector_store.create_vector_store_from_json(
    #     json_file_path, vector_store_path
    # )
    # print(f"Vector store created with {total_entries} entries and {total_added_docs} documents added.")
    app.run(debug=True)