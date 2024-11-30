import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st

# Configurar a chave de API do OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Funções para extrair texto de PDF e gerar embeddings
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_embeddings(text):
    sentences = text.split(". ")
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings

def store_embeddings(embeddings, sentences):
    vectors = [(str(i), embeddings[i].tolist(), {"text": sentences[i]}) for i in range(len(sentences))]
    return vectors

def query_assistant(question, all_sentences, all_embeddings):
    # Geração do embedding da pergunta
    query_embedding = embedding_model.encode([question])[0]

    # Simulação de similaridade com embeddings
    from numpy import dot
    from numpy.linalg import norm

    similarities = [
        dot(query_embedding, emb) / (norm(query_embedding) * norm(emb))
        for emb in all_embeddings
    ]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]

    # Construir contexto com as sentenças mais relevantes
    context = " ".join([all_sentences[i] for i in top_indices])

    # Gerar a resposta usando o OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente inteligente."},
            {"role": "user", "content": f"Contexto: {context}\n\nPergunta: {question}"}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"]

# Interface com Streamlit
st.title("Assistente baseado em Múltiplos PDFs")
st.write("Carregue vários arquivos PDF e faça perguntas baseadas no conteúdo!")

# Upload de múltiplos PDFs
uploaded_files = st.file_uploader("Envie os arquivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_sentences = []
    all_embeddings = []
    
    with st.spinner("Processando os arquivos PDF..."):
        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            sentences, embeddings = generate_embeddings(text)
            
            # Armazenar todas as sentenças e embeddings
            all_sentences.extend(sentences)
            all_embeddings.extend(embeddings)

    st.success("Arquivos processados! Agora, você pode fazer perguntas.")
    question = st.text_input("Digite sua pergunta:")

    if question:
        with st.spinner("Gerando resposta..."):
            answer = query_assistant(question, all_sentences, all_embeddings)
            st.write("Resposta:", answer)
