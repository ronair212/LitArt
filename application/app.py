import sys
import os
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie, st_lottie_spinner
import json

from scripts.inference import summarize,generate
from LLama2.scripts.inference import generate_response
from utilities.helper_functions import text_to_prompt 


st.set_page_config(page_title="Book covers", page_icon=":notebook_with_decorative_cover:")
processing_done = False

st.sidebar.title("Navigation")    
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload Chapters", "Generate Book Covers"])


def load_lottie_file(path:str):
    with open(path) as f:
        data = json.load(f)
    return data

def generate_text(temperature,chapter:str,sample:bool):
    lottie = load_lottie_file("../utilities/Image_load.json")
    with st_lottie_spinner(lottie,height=200):
        summary_text = generate_response(chapter=chapter,
                                model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13",
                                max_new_tokens = 500,
                                do_sample = sample,
                                temperature = float(temperature),
                                top_p = 0.8,
                                )
        st.success("Summary created successfully!")
    return summary_text

def generate_image(prompt:str,book_title:str,adapter:str,num_covers:int=1):
    lottie = load_lottie_file("../utilities/Image_gen.json")
    with st_lottie_spinner(lottie,height=300):
        for i in range(num_covers):
            book_cover = generate(prompt=prompt,
                model_name="CompVis/stable-diffusion-v1-4",
                file_name=book_title+f"_{time.time()}",
                lora=f"/work/LitArt/adwait/capstone/trained_adapters/{adapter}_1.1.0/")

        st.subheader("Generated Book Covers")
        col1,col2 = st.columns(2)
        with col1:
            st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt for dune using the suspense adapter")
        with col2:
            st.image("../../sample_outputs/Llama_dune_1711161535.376433.png", caption="Prompt")
    

if app_mode == "Upload Chapters":
    st.subheader("Upload Your Chapter Here")
    summarizer = st.selectbox('Choose the model to be used for summarization',['LLama','BART','T5','Seq2Seq'])
    if summarizer == 'LLama':
        temprature = st.select_slider('Choose temprature',[0.5,0.6,0.7,0.8,0.9])
        sample = st.selectbox('Sample',[True,False])

    file_type = st.selectbox("Select File Type", ("pdf", "txt", "plain text"))

    if file_type in ("pdf", "txt"):
        uploaded_file = st.file_uploader("Choose a file")  
        if uploaded_file is not None:
            with open(uploaded_file) as f:
                chapter = f.read()  
                summary_text = generate_text(temperature=temprature,
                                             chapter=chapter,
                                             sample=sample)
                st.text(f"{summary_text}")
                processing_done = True 
    
    else:
        text_input = st.text_area("Write your text here")
        if text_input:  
            lottie = load_lottie_file("../utilities/anime.json")
            with st_lottie_spinner(lottie,height=200):
                summary_text = generate_response(chapter=text_input,
                                        model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13",
                                        max_new_tokens = 500,
                                        do_sample = sample,
                                        temperature = int(temperature),
                                        top_p = 0.8,
                                        )
                st.success("Summary created successfully!")
                st.text(f"{summary_text}")
                processing_done = True 


if app_mode == "Generate Book Covers":
    st.subheader("Generate Book Covers")
    st.text_area("Enter your prompt here keep it under 50 words ")
    button_clicked = st.button("Generate Book Covers")
    generate_image(prompt=summary_text)

if processing_done:
    button_clicked = st.button("Generate Book Covers")

    if button_clicked:
        generate_image(prompt=summary_text)