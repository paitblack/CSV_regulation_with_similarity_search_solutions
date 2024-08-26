import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from datasketch import MinHash, MinHashLSH
import chardet

def get_csv_col_as_a_list(csv, col_name):   #get the column as a list.
    df = pd.read_csv(csv)
    return df[col_name].tolist()

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

def get_minhash(text):        #initialize the minhash
    minhash = MinHash()
    for word in text.split():  #split the text into individuals.
        minhash.update(word.encode('utf8'))   # update the minhash object with utf8 encoded version.
    return minhash

def detect_file_encoding(file_path):   #detects the encoding of a file, which is useful for reading CSV files that might not be encoded in UTF-8.
    with open(file_path, 'rb') as f:    # read the file in binary mode. (rb = read binarey)
        result = chardet.detect(f.read())   
    return result['encoding']

def process_csv(input_csv, col1, col2, output_csv, threshold=0.30):   # select the threshold by yourself.
    encoding = detect_file_encoding(input_csv)   # detect encoding.
    df = pd.read_csv(input_csv, encoding=encoding)
    
    col1_data = preprocess_data(df[col1].tolist())   #choose 2 columns you want to search similarly.
    col2_data = preprocess_data(df[col2].tolist())     # you can add any other columns or delete one of the columns if needed.
    
    lsh = MinHashLSH(threshold=threshold)
    minhashes = []
    
    for i in range(len(col1_data)):             # create MinHash objects for each row.
        minhash1 = get_minhash(col1_data[i])
        minhash2 = get_minhash(col2_data[i])
        combined_minhash = MinHash()
        combined_minhash.update(minhash1.digest())
        combined_minhash.update(minhash2.digest())
        lsh.insert(i, combined_minhash)
        minhashes.append(combined_minhash)
    
    to_remove = set()
    for i in range(len(minhashes)):               #identift similar rows here.
        if i in to_remove:
            continue
        query_minhash = minhashes[i]
        results = lsh.query(query_minhash)
        results = [r for r in results if r != i]
        if results:
            to_remove.update(results)
    
    filtered_df = df.loc[~df.index.isin(to_remove)]    # filter out similar rows
    filtered_df.to_csv(output_csv, index=False)