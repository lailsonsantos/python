# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:52:28 2020

@author: lailson
"""

# CLI para build
import streamlit as st

# Text/Title
st.title("Título da página")

# Header/Subheader
st.header("Header")
st.subheader("Subheader")

# Text
st.text("Hello world")

#  Markdown
st.markdown("## Markdown")

# Mensagens ao usuário
st.success("Mensagem de Sucesso")
st.info("Mensagem de informação")
st.warning("Mensagem de aviso")
st.error("Mensagem de erro")
st.exception("Nome do erro('Erro ao logar')")

# Pega a informação sobre o Python
st.help(range)

# Escreve texto
st.write("Texto escrito")
st.write(range(10))

# Imagens
from PIL import Image
img = Image.open("imagem.jpeg")
st.image(img, width=300, caption="Rotulo da imagem")

# Videos
video_file = open("video.mp4", "rb").read()
# video_bytes = video_file.read()
st.video(video_file)

# Audio
#audio_file = open("audio.mp3", "rb").read()
#st.audio(audio_file, format='audio/mp3')

# Widgets
# Checkbox
if st.checkbox("Show/Hide"):
    st.text("Mostrar ou esconder widgets")

# Radio
status = st.radio("Qual o estado", ("Ativo", "Inativo"))

if status == "Ativo":
    st.success("Está ativo")
else:
    st.warning("Está inativo")

# SelectBox
trabalho = st.selectbox("Qual a sua profissão", ["Programador", "Designer"])
st.write("A profissao selecionada: ", trabalho)

# MultiSelect
local = st.multiselect("Aonde você trabalha?", ("Londres", "Estados Unidos", "Brasil"))
st.write("Você trabalha: ", len(local), "aqui")

# Slides
