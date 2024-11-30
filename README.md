# TRABALHO-TOPICOS
# Assistente de Perguntas Baseado em Múltiplos PDFs

Este projeto utiliza **Streamlit**, **OpenAI** e **SentenceTransformers** para criar um assistente capaz de responder perguntas com base no conteúdo de múltiplos arquivos PDF. O sistema processa os PDFs, extrai seu conteúdo e usa inteligência artificial para responder às perguntas do usuário.

---## Recursos

- Upload de múltiplos arquivos PDF.
- Extração de texto e criação de embeddings.
- Consulta inteligente utilizando o modelo GPT da OpenAI.
- Interface intuitiva desenvolvida com Streamlit.

---## Recursos

- Upload de múltiplos arquivos PDF.
- Extração de texto e criação de embeddings.
- Consulta inteligente utilizando o modelo GPT da OpenAI.
- Interface intuitiva desenvolvida com Streamlit.
  Link para utilização pelo stremlit
https://geccgtsvrk8mkwdcgdq9me.streamlit.app/


para poder rodar localmente
tem que fazer dessa forma com os determinados comentarios

import os

#OPENAI_API_KEY = st.secrets["OPENAI_API_KEY

INDEX_NAME = "multilingual-e5-large"

# Configurar a chave de API do OpenAI

openai.api_key = "coloque sua key"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
