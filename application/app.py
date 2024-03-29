import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie, st_lottie_spinner
import json

st.set_page_config(page_title="Book covers", page_icon=":notebook_with_decorative_cover:")
processing_done = False

st.sidebar.title("Navigation")    
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload Chapters", "Generate Book Covers"])

def load_lottie_file(path:str):
    with open(path) as f:
        data = json.load(f)
    return data

if app_mode == "Upload Chapters":
    st.subheader("Upload Your Chapter Here")
    summarizer = st.selectbox('Choose the model to be used for summarization',['LLama','BART','T5','Seq2Seq'])
    if summarizer == 'LLama':
        temprature = st.select_slider('Choose temprature',[0,1,2,3])
    file_type = st.selectbox("Select File Type", ("pdf", "txt", "plain text"))

    if file_type in ("pdf", "txt"):
        uploaded_file = st.file_uploader("Choose a file")  
        if uploaded_file is not None:   
            lottie = load_lottie_file("../utilities/Image_load.json")
            with st_lottie_spinner(lottie,height=200):
                time.sleep(5)
                st.success("File processed successfully!")
                processing_done = True 
    
    else:
        text_input = st.text_area("Write your text here")
        if text_input:  
            lottie = load_lottie_file("../utilities/anime.json")
            with st_lottie_spinner(lottie,height=200):
                time.sleep(5)
                st.success("Text processed successfully!")
                processing_done = True 

if app_mode == "Generate Book Covers":
    st.subheader("Generate Book Covers")
    st.text_area("Enter your prompt here keep it under 50 words ")
    button_clicked = st.button("Generate Book Covers")
    if button_clicked:
        lottie = load_lottie_file("../utilities/Image_gen.json")
        with st_lottie_spinner(lottie,height=300):
            time.sleep(5) 
            st.subheader("Generated Book Covers")
            col1,col2 = st.columns(2)
            with col1:
                st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt for dune using the suspense adapter")
            with col2:
                st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt")

if processing_done:
    button_clicked = st.button("Generate Book Covers")

    if button_clicked:
        lottie = load_lottie_file("../utilities/Image_gen.json")
        with st_lottie_spinner(lottie,height=300):
            time.sleep(5) 
            st.subheader("Generated Book Covers")
            col1,col2 = st.columns(2)
            with col1:
                st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt for dune using the suspense adapter")
            with col2:
                st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt")