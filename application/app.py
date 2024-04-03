import sys
import os
sys.path.insert(1, os.path.expanduser('~') + '/LitArt/')

import streamlit as st
import time
import requests
from streamlit_lottie import st_lottie, st_lottie_spinner
import json
import PyPDF2
import io
import torch

from scripts.inference import summarize,generate
from LLama2.scripts.inference import generate_response
from utilities.helper_functions import text_to_prompt
from T5.scripts.t5_inference import summarize_t5
from Pegasus.scripts.pegasus_inference import summarize_pegasus
from BART.scripts.bart_inference import summarize_bart


st.set_page_config(page_title="Book covers", page_icon=":notebook_with_decorative_cover:")
processing_done = False
image_generated = False
summarizers = {'T5': summarize_t5
              ,'BART':summarize_bart
              ,'Pegasus':summarize_pegasus}

st.sidebar.title("Navigation")    
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload Chapters", "Generate Book Covers"])

def load_lottie_file(path:str):
    with open(path) as f:
        data = json.load(f)
    return data

@st.cache_data(show_spinner=False)
def generate_text(chapter:str,sample:bool=False,temperature:int=1,summarizer:str="LLama"):
    if summarizer=="LLama":
        summary_text = generate_response(chapter=chapter,
                            model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13",
                            max_new_tokens = 500,
                            do_sample = sample,
                            temperature = float(temperature),
                            top_p = 0.8,
                            )
    else:
        summary_text = summarizers[summarizer](chapter)

    st.success("Summary created successfully!")
    torch.cuda.empty_cache()
    return summary_text

@st.cache_data(show_spinner=False)
def generate_image(prompt:str,book_title:str,adapter:str,num_covers:int=1):
    covers = []
    for i in range(num_covers):
        book_cover = generate(prompt=prompt,
            model_name="CompVis/stable-diffusion-v1-4",
            file_name=book_title+f"_{time.time()}",
            lora=f"/work/LitArt/adwait/capstone/trained_adapters/{adapter}_1.1.0/")
        
        covers.append(book_cover)

    st.subheader("Generated Book Covers")
    cols = st.columns(num_covers)

    for i, image in enumerate(covers):
        with cols[i]:
            st.image(image, caption=prompt)

def clear_cache():
    st.cache_data.clear()

if app_mode == "Upload Chapters":
    st.subheader("Upload Your Chapter Here")
    summarizer = st.selectbox('Choose the model to be used for summarization',['LLama','BART','T5','Seq2Seq'])
    if summarizer == 'LLama':
        temprature = st.select_slider('Choose temprature',[0.5,0.6,0.7,0.8,0.9,1])
        sample = st.selectbox('Sample',["True","False"])



    file_type = st.selectbox("Select File Type", ("pdf", "txt", "plain text"))

    if file_type == "txt":
        uploaded_file = st.file_uploader("Choose a file")
        lottie = load_lottie_file("../utilities/anime.json")
        if uploaded_file is not None:  
            chapter = uploaded_file.getvalue().decode('utf-8')
            st.write(chapter)
            with st_lottie_spinner(lottie,height=200):
                summary_text = generate_text(temperature=temprature,
                                            chapter=chapter,
                                            sample=sample)
            processing_done = True 

    elif file_type == 'pdf':
        uploaded_file = st.file_uploader("Choose a file")
        lottie = load_lottie_file("../utilities/anime.json")
        if uploaded_file is not None:
            content = uploaded_file.getvalue()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            chapter = ''
            
            for page_num in range(len(pdf_reader)):
                page = pdf_reader.getPage(page_num)
                chapter += page.extractText()

            with st_lottie_spinner(lottie,height=200):
                summary_text = generate_text(temperature=temprature,
                                            chapter=chapter,
                                            sample=sample)
            processing_done = True 

    else:
        text_input = st.text_area("Write your text here")
        if text_input is not None:
            lottie = load_lottie_file("../utilities/anime.json")
            with st_lottie_spinner(lottie,height=200):
                summary_text = generate_text(temperature=temprature,
                                        chapter=text_input,
                                        sample=sample)
                st.success("Summary created successfully!")

                processing_done = True 
if summary_text:
    st.subheader("Generated Summary")
    st.write(f"{summary_text}")

if app_mode == "Generate Book Covers":
    st.subheader("Generate Book Covers")
    lottie = load_lottie_file("../utilities/Image_gen.json")

    adapter = st.selectbox("Select an adapter",['fiction','fantasy','suspense','speculative_fiction'])
    title = st.text_input("Enter book title")
    num_covers = st.select_slider("Number of Book covers",[1,2,3])
    prompt = st.text_area("Enter your prompt here keep it under 30 words ")
    prompt = text_to_prompt(text=prompt,model=adapter)

    button_clicked = st.button("Generate Book Covers")

    if button_clicked:
        with st_lottie_spinner(lottie,height=200):
            generate_image(prompt=prompt,
                    book_title=title,
                    adapter=adapter,
                    num_covers=int(num_covers))
            
        torch.cuda.empty_cache()
        image_generated = True


if processing_done :
    adapter = st.selectbox("Select an adapter",['fiction','fantasy','suspense','speculative_fiction'])
    title = st.text_input("Enter book title")
    num_covers = st.select_slider("Number of Book covers",[1,2,3])
    button_clicked = st.button("Generate Book Covers")
    lottie = load_lottie_file("../utilities/Image_gen.json")
    
    if button_clicked:
        prompt = text_to_prompt(text=summary_text,model=adapter)
        with st_lottie_spinner(lottie,height=200):
            generate_image(prompt=prompt,
                    book_title=title,
                    adapter=adapter,
                    num_covers=int(num_covers))
        
        torch.cuda.empty_cache()
        image_generated = True

if image_generated:
    clear_cache = st.button("clear cache")
    if clear_cache:
        clear_cache()