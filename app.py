"""
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.report_analysis import analyze_report

app = Flask(__name__)

load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Using the new HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create an instance of Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "quickstart"

# Check if the index exists and create/load it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_API_ENV
        )
    )

# Access the existing index
index = pc.Index(index_name)

# Initialize the Pinecone vector store from the existing index
docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    #config={'max_new_tokens': 512, 'temperature': 0.8}
    config={'max_new_tokens': 256, 'temperature': 0.8}

)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Analyze the uploaded report
    analysis_result = analyze_report(file_path)

    return jsonify({"result": analysis_result})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

"""


from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from pinecone import Pinecone, ServerlessSpec
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from src.report_analysis import analyze_report, process_large_text_with_chunking
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

load_dotenv()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Using the new HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create an instance of Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "quickstart"

# Check if the index exists and create/load it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_API_ENV
        )
    )

# Access the existing index
index = pc.Index(index_name)

# Initialize the Pinecone vector store from the existing index
docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    #config={'max_new_tokens': 512, 'temperature': 0.8}
    config={'max_new_tokens': 256, 'temperature': 0.8}

)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Analyze the uploaded report
    analysis_result = analyze_report(file_path)

    # Chunk the analysis result to fit within the model's token limit
    chunk_size = 500  # Adjust the size based on model token limit and overlap
    chunks = [analysis_result[i:i+chunk_size] for i in range(0, len(analysis_result), chunk_size)]

    # Process each chunk and combine the responses
    responses = []
    for chunk in chunks:
        response = qa({"query": chunk})
        responses.append(response["result"])
    
    # Combine responses
    combined_response = "\n".join(responses)

    print("Response:", combined_response)
    return combined_response




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



