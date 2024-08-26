import pandas as pd
import numpy as np
import re
import faiss
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_ready_with_columns(csv, co1, co2):
    df = pd.read_csv(csv)
    return df[[co1, co2]].apply(lambda x: ' '.join(x.dropna()), axis=1).tolist()

def text_preprocessor(text):     # preprocess the text to get correct outputs.
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_data(data_list):   # preprocess every element of a list.
    return [text_preprocessor(text) for text in data_list]

def vectorize_data(data_list):  # vectorize texts with TF-IDF.
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data_list).toarray()
    return vectors

def find_similar_rows(vectors, threshold=0.8):  #find similar rows with faiss.
    d = vectors.shape[1]                            #select best threshold by yourself.
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    D, I = index.search(vectors, 5)  #find most similar 5.
    
    similar_pairs = {}
    for i, (dist, idx) in enumerate(zip(D, I)):
        if dist[1] < threshold: #sign similar rows to drop.
            if idx[1] > i: 
                similar_pairs[idx[1]] = i
    
    return similar_pairs

def deduplicate_rows(df, similar_pairs):   #drop duplicate rows.
    drop_indices = set(similar_pairs.keys())
    deduplicated_df = df.drop(index=drop_indices)
    return deduplicated_df

def main(csv_file, co1, co2, threshold=0.8):
    combined_columns = get_ready_with_columns(csv_file, co1, co2)
    preprocessed_data = preprocess_data(combined_columns)
    vectors = vectorize_data(preprocessed_data)
    similar_pairs = find_similar_rows(vectors, threshold=threshold)
    
    df = pd.read_csv(csv_file)
    deduplicated_df = deduplicate_rows(df, similar_pairs)
    
    return deduplicated_df

"""
In similar rows that we find in csv, one may contain more information than the other.
In order to get an efficient result and get more information, this method can be added to merge these csv rows appropriately.

def merge_similar_rows(df, similar_pairs, co1, co2):  
    for i, j in similar_pairs.items():
        row_i = df.loc[i]
        row_j = df.loc[j]
        
        if row_i.count() > row_j.count():
            df.loc[i] = row_i.combine_first(row_j) 
            df = df.drop(index=j)
        else:
            df.loc[j] = row_j.combine_first(row_i)
            df = df.drop(index=i)
    
    return df
"""