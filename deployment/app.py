import sys
import os

# Project path
project_path = '/work/LitArt/patel/LitArt/'
sys.path.insert(1, project_path)

from flask import Flask, render_template, request
from scripts.inference import generate, summarize
from LLama2.scripts.inference import generate_response
from data_module.data_preprocessor import TextPreprocessing
from utilities.helper_functions import text_to_prompt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        # Extract text from the form submission
        chapter_text = request.form.get('text', '')

        print(chapter_text)
        
        # Initialize the TextPreprocessing class
        preprocessor = TextPreprocessing(slang=False, stopwordList=None, stemming=False, lemmatization=False)
        
        # Preprocess the text
        preprocessed_text = preprocessor.process(chapter_text)
        
        # Generate summary using the preprocessed text
        # summary_text = generate_response(chapter=preprocessed_text, model_dir="/work/LitArt/nair/outdir/meta-llama-Llama-2-7b-hf-2024-03-21-14:17:13")

        summary_text = summarize(chapter=chapter_text)

        prompt = text_to_prompt(text=summary_text)
        
        # Function to generate an image based on the summary
        image_file = generate(prompt=prompt, file_name='generated_image')
        
        # Return summary and image to the user
        return render_template('index.html', summary_text=prompt, image_file=image_file)
    else:
        # Handle GET request by showing the form without any summary
        return render_template('index.html', summary_text=None, image_file=None)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
