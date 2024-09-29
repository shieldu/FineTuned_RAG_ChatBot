from flask import Flask, request, jsonify, render_template
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# ChromaDB 클라이언트 설정
client = chromadb.Client()
collection = client.create_collection("security_manual")

# 파인튜닝된 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
model = AutoModelForCausalLM.from_pretrained("kingkim/kodialogpt_v3.0_SecurityManual")
gpt2_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# SentenceTransformer 모델 로드 (임베딩 모델을 더 성능이 좋은 paraphrase-multilingual로 교체)
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 1. PDF 업로드 및 처리 (문맥을 포함한 임베딩을 위해 세 문장씩 묶음)
def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    sentences = text.split('.')
    grouped_sentences = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return grouped_sentences

# 2. 문장 임베딩 후 ChromaDB에 저장
def store_in_chroma(sentences):
    embeddings = sentence_model.encode(sentences)
    ids = [f"id_{i}" for i in range(len(sentences))]  # 각 문장에 대한 고유 ID 생성
    for i, sentence in enumerate(sentences):
        collection.add(ids=[ids[i]], embeddings=[embeddings[i]], metadatas={"sentence": sentence})

# 3. 질문 인코딩 및 유사 문장 찾기 (상위 3개의 결과를 결합하여 사용)
def find_similar_sentences(question, top_n=3):
    question_embedding = sentence_model.encode([question])
    results = collection.query(query_embeddings=question_embedding, n_results=top_n)
    
    similar_sentences = []
    if "metadatas" in results:
        for metadata in results["metadatas"]:
            similar_sentences.append(metadata[0].get("sentence", ""))
    
    return ' '.join(similar_sentences)  # 가장 유사한 문장들을 결합하여 반환

# 4. GPT-2 모델로 답변 생성 (질문을 포함한 프롬프트 추가)
def generate_response(similar_sentence, question):
    prompt = f"질문: {question}\n유사한 문장: {similar_sentence}\n답변:"
    generation_args = {
        "max_new_tokens": 80,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }
    response = gpt2_pipeline(prompt, **generation_args)
    return response[0]['generated_text']

# 5. Flask 라우팅 설정
@app.route('/')
def home():
    return render_template('tuned.html')

@app.route('/eco_upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    if file:
        file.save('uploaded_pdf.pdf')
        sentences = process_pdf('uploaded_pdf.pdf')
        store_in_chroma(sentences)
        return jsonify({"message": "PDF 업로드 및 처리 완료"}), 200
    return jsonify({"message": "파일 업로드 실패"}), 400

@app.route('/eco_ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if question:
        similar_sentence = find_similar_sentences(question)
        answer = generate_response(similar_sentence, question)
        return jsonify({"answer": answer}), 200
    return jsonify({"answer": "질문을 입력해주세요"}), 400

if __name__ == '__main__':
    app.run(debug=True)

