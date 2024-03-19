import pandas as pd
import re
import requests
from bs4 import BeautifulSoup

file_path = 'booksummaries.txt'
output_file_path = 'CMU_df.csv'

column_names = ['ID', 'URI', 'Title', 'Author', 'Publication Date', 'Genres', 'Synopsis']

def convert_to_goodreads_url(book_titles):
    base_url = "https://www.goodreads.com/search?utf8=%E2%9C%93&q="
    search_type = "&search_type=books"
    
    def preprocess_title(title):
        processed_title = re.sub(r'[^a-zA-Z0-9 ]', '', title)
        processed_title = processed_title.replace(' ', '+')
        return processed_title
    
    urls = []
    for title in book_titles:
        processed_title = preprocess_title(title)
        full_url = f"{base_url}{processed_title}{search_type}"
        urls.append(full_url)
    
    return urls

def new_urls(urls):
    new_urls = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                html_content = response.text
                soup = BeautifulSoup(html_content, 'html.parser')
                book_title_element = soup.find('a', class_='bookTitle')
                if book_title_element is not None and 'href' in book_title_element.attrs:
                    book_title_url = 'https://www.goodreads.com' + book_title_element['href']
                    new_urls.append(book_title_url)
                else:
                    new_urls.append("None")
            else:
                new_urls.append("None")
        except Exception as e:
            new_urls.append("None")
    return new_urls
import pandas as pd

output_file_path = 'CMU_df.csv'

def process_books_in_chunks(chunk_size=100):
    try:
        CMU_df = pd.read_csv(output_file_path)
    except FileNotFoundError:
        file_path = 'booksummaries.txt'
        column_names = ['ID', 'URI', 'Title', 'Author', 'Publication Date', 'Genres', 'Synopsis']
        CMU_df = pd.read_csv(file_path, sep='\t', names=column_names, parse_dates=['Publication Date'])
        CMU_df['URL'] = None  

    unprocessed_df = CMU_df[CMU_df['URL'].isna()]

    if unprocessed_df.empty:
        print("All titles have been processed.")
        return

    remaining_titles = unprocessed_df['Title'].tolist()
    
    for i in range(0, len(remaining_titles), chunk_size):
        chunk_titles = remaining_titles[i:i+chunk_size]
        chunk_urls = convert_to_goodreads_url(chunk_titles)
        chunk_urls_new = new_urls(chunk_urls)
        
        for j, title in enumerate(chunk_titles):
            indices = CMU_df[CMU_df['Title'] == title].index
            for index in indices:
                CMU_df.loc[index, 'URL'] = chunk_urls_new[j]
        
        CMU_df.to_csv(output_file_path, index=False)
        print(f"Processed {min(i + chunk_size, len(remaining_titles))} of {len(remaining_titles)} titles.")
        print(f"DataFrame saved to '{output_file_path}'")


process_books_in_chunks()


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.parse import unquote

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
                  "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
                  "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                  "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
                  "with", "about", "against", "between", "into", "through", "during", "before", "after",
                  "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
                  "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
                  "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
                  "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
                  "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
                  "wasn", "weren", "won", "wouldn"}

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def extract_book_name_from_url(url):
    url = unquote(url)
    match = re.search(r"goodreads.com/book/show/\d+\.([-\w]+)", url)
    if match:
        book_name_part = match.group(1)
        book_name_part = book_name_part.replace('_', ' ').replace('-', ' ')
        return preprocess_text(book_name_part)
    return []

def process_books_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    
    df = df[df['URL'] != "None"]
    
    df['Title_Words'] = df['Title'].apply(preprocess_text)
    df['URL_Words'] = df['URL'].apply(extract_book_name_from_url)
    
    df_filtered = df[df.apply(lambda x: len(set(x['Title_Words']) & set(x['URL_Words'])) > 0, axis=1)]
    
    df_filtered = df_filtered.drop(columns=['Title_Words', 'URL_Words'])
    
    df_filtered.to_csv(output_csv_path, index=False)
    
    return df_filtered

input_csv_path = 'CMU_df.csv'  
output_csv_path = 'CMU_df_processed.csv'  

process_books_csv(input_csv_path, output_csv_path)

import pandas as pd

cmu_data_with_images_df = pd.read_csv('/work/LitArt/data/filtered_CMU_data_with_images.csv')


import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tqdm
import re
import urllib.request

images_dir = "/work/LitArt/data/images"
os.makedirs(images_dir, exist_ok=True)

CMU_df = pd.read_csv('/work/LitArt/data/filtered_CMU_data_with_images.csv')  # Update this path if necessary

if "isImageAvailable" not in CMU_df.columns:
    CMU_df["isImageAvailable"] = False
if "imageLocation" not in CMU_df.columns:
    CMU_df["imageLocation"] = ""

def fetch_images_in_chunks(chunk_size=10):
    for i in tqdm.tqdm(range(0, len(CMU_df), chunk_size), desc="Fetching images"):
        chunk = CMU_df.iloc[i:i+chunk_size]
        for index, row in chunk.iterrows():
            # Skip rows where the image is already downloaded
            if row["isImageAvailable"] and os.path.isfile(row["imageLocation"]):
                continue

            book = row["Title"]
            author = row["Author"] if pd.notna(row["Author"]) else "Unknown"
            link = row["URL"]

            filename = f"{book}-{author}".replace(' ', '_')
            filename = re.sub(r'[^a-zA-Z0-9_-]', '', filename)
            image_path = os.path.join(images_dir, f"{filename}.jpg")

            if not os.path.isfile(image_path) and pd.notnull(link):
                try:
                    response = requests.get(link)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    imageLink = soup.find("div", {"class": "BookCover__image"})
                    if imageLink:
                        imageLink = imageLink.select("img[src^=http]")[0]["src"]
                        urllib.request.urlretrieve(imageLink, image_path)
                        CMU_df.at[index, "isImageAvailable"] = True
                        CMU_df.at[index, "imageLocation"] = image_path
                    else:
                        print(f"Image not found for {book} by {author}")
                except Exception as e:
                    print(f"Exception for {book} by {author}: {repr(e)}")
            elif os.path.isfile(image_path):
                CMU_df.at[index, "isImageAvailable"] = True
                CMU_df.at[index, "imageLocation"] = image_path

        CMU_df.to_csv("/work/LitArt/data/filtered_CMU_data_with_images.csv", index=False)

fetch_images_in_chunks()


