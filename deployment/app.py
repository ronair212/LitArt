from flask import Flask, render_template, request
import os
import sys
import fitz
import torch
from datetime import datetime

# Define project path
project_path = '/work/LitArt/patel/LitArt/'
sys.path.insert(1, project_path)

from scripts.inference import generate  
from LLama2.scripts.inference import generate_response  
from data_module.data_preprocessor import TextPreprocessing  
from utilities.helper_functions import text_to_prompt  

app = Flask(__name__)

# Adjust as per your environment
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directory paths for data, summaries, and generated images
data_dir = os.path.join(project_path, 'deployment/data')
prompt_dir = os.path.join(project_path, 'deployment/generated_prompt')
image_dir = os.path.join(app.static_folder, 'deployment/static/generated_images')

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(prompt_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def LitArt():
    if request.method == 'POST':
        file = request.files.get('file')
        chapter_text = ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file and file.filename != '':
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join(data_dir, filename)
            file.save(filepath)
            # Process the uploaded file
            if file.filename.endswith('.pdf'):
                doc = fitz.open(filepath)
                chapter_text = " ".join(page.get_text() for page in doc)
            elif file.filename.endswith('.txt'):
                with open(filepath, 'r') as f:
                    chapter_text = f.read()
        else:
            # Use manually entered text
            chapter_text = request.form.get('text', '')
            # Save the entered text
            with open(os.path.join(data_dir, f"{timestamp}_text.txt"), 'w') as f:
                f.write(chapter_text)

        # Preprocess the text
        preprocessor = TextPreprocessing(slang=False, stopwordList=None, stemming=False, lemmatization=False)
        preprocessed_text = preprocessor.process(chapter_text)

        # Determine which model was chosen based on the button pressed
        model_choice = request.form.get('model')
        summary_text = ""
        if model_choice == 'llama':
            summary_text = generate_response(chapter=preprocessed_text, model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13")
        elif model_choice == 'pegasus':
            # Specify the Google Pegasus model name directly or use a variable
            summary_text = summarize(chapter=preprocessed_text, model_name="google/pegasus-xsum")

        # Generate a prompt from the summary
        prompt = text_to_prompt(text=summary_text)
        
        # Save the generated prompt
        prompt_filename = f"{timestamp}_prompt.txt"
        with open(os.path.join(prompt_dir, prompt_filename), 'w') as f:
            f.write(prompt)

        # Generate and save the image based on the summary
        image_filename = generate(prompt=prompt, file_name=f"{timestamp}_image")
        image_path = os.path.join('generated_images', image_filename)

        # Return summary and image to the user
        return render_template('index.html', summary_text=summary_text, image_file=image_path)
    else:
        # Handle GET request by showing the form without any summary or image
        return render_template('index.html', summary_text=None, image_file=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
