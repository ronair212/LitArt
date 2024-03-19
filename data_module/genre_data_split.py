import pandas as pd
from datasets import load_dataset, load_from_disk
import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
from datasets import Dataset, DatasetDict
import re



df = pd.read_csv("/work/LitArt/data/filtered_CMU_data_with_images.csv")

df = df.dropna(subset=['Genres'])


def transform_genre(genre_col):
    transformed = []
    for item in genre_col:
        genre_dict = eval(item)
        genres = ', '.join(list(genre_dict.values()))
        transformed.append(genres)
    return transformed

df['Genres'] = transform_genre(df['Genres'])

def genre_split_dic(genre_list):
    genre_dic = {}
    for i , row in enumerate(genre_list):
        genres = row.split(', ')
        for gen in genres:
            if gen not in genre_dic.keys():
                genre_dic[gen] = [i]
            else:
                genre_dic[gen].append(i)
    return genre_dic

genre_dic = genre_split_dic(df['Genres'])

#print("the count if unique genre is " , len(genre_dic.keys()))

p = [row.split(', ') for row in df['Genres']]

def flatten(xss):
    return [x for xs in xss for x in xs]
l = flatten(p)

l = set(l)


def create_subsets(df, dicto):
    subsets = {}
    for key, indices in dicto.items():
        subsets[key] = df.iloc[indices]
    return subsets

subsets = create_subsets(df, genre_dic)

for key, subset_df in subsets.items():
    filename_safe = key.replace(' ', '_').replace('à', 'a') + '.csv'
    subset_df.to_csv(f'/work/LitArt/data/genre_split_CMU/{filename_safe}', index=False)

[f'/work/LitArt/data/genre_split_CMU/{key.replace(" ", "_").replace("à", "a")}.csv' for key in subsets.keys()]



