import pandas as pd
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import re

## Levenshtein Distance

# clean data
def clean_data(df):
    # remove non-alphanumeric characters
    df['NO_IDENT'] = df['NO_IDENT'].str.replace(r'\W', ' ', regex=True)
    df['NM_PENERIMA'] = df['NM_PENERIMA'].str.replace(r'\W', ' ', regex=True)
    df['AL_PENERIMA'] = df['AL_PENERIMA'].str.replace(r'\W', ' ', regex=True)
    df['URAIAN_BARANG'] = df['URAIAN_BARANG'].str.replace(r'\W', ' ', regex=True)
    # remove word with length less than 3
    df['NO_IDENT'] = df['NO_IDENT'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    df['NM_PENERIMA'] = df['NM_PENERIMA'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    df['AL_PENERIMA'] = df['AL_PENERIMA'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    df['URAIAN_BARANG'] = df['URAIAN_BARANG'].str.replace(r'\b\w{1,2}\b', '', regex=True)
    # remove multiple spaces
    df['NO_IDENT'] = df['NO_IDENT'].str.replace(r'\s+', ' ', regex=True)
    df['NM_PENERIMA'] = df['NM_PENERIMA'].str.replace(r'\s+', ' ', regex=True)
    df['AL_PENERIMA'] = df['AL_PENERIMA'].str.replace(r'\s+', ' ', regex=True)
    df['URAIAN_BARANG'] = df['URAIAN_BARANG'].str.replace(r'\s+', ' ', regex=True)
    # remove leading and trailing spaces
    df['NO_IDENT'] = df['NO_IDENT'].str.strip()
    df['NM_PENERIMA'] = df['NM_PENERIMA'].str.strip()
    df['AL_PENERIMA'] = df['AL_PENERIMA'].str.strip()
    df['URAIAN_BARANG'] = df['URAIAN_BARANG'].str.strip()
    # lowercase
    df['NO_IDENT'] = df['NO_IDENT'].str.lower()
    df['NM_PENERIMA'] = df['NM_PENERIMA'].str.lower()
    df['AL_PENERIMA'] = df['AL_PENERIMA'].str.lower()
    df['URAIAN_BARANG'] = df['URAIAN_BARANG'].str.lower()
    # remove nan uraian barang
    df = df.dropna(subset=['URAIAN_BARANG'])
    # reset index
    df.reset_index(drop=True, inplace=True)
    return df
  
# Load and preprocess the data
def load_data(filepath):
    df = pd.read_csv(filepath, dtype=str)
    df = df[['ID_AJU','NO_IDENT', 'NM_PENERIMA', 'AL_PENERIMA','HS_CODE','URAIAN_BARANG','JML_SAT_HRG','CIF_DETAIL']]
    # df = df[['ID_AJU','NO_IDENT', 'NM_PENERIMA', 'AL_PENERIMA','HS_CODE','URAIAN_BARANG']]
    df = clean_data(df)
    return df

# Create an index for searching
def create_index(data):
    vectorizer_ident = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1,3))
    vectorizer_name = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1,3))
    vectorizer_address = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1,3))
    vectorizer_uraian = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1,3))
    
    tfidf_matrix_ident = vectorizer_ident.fit_transform(data['NO_IDENT'])
    tfidf_matrix_name = vectorizer_name.fit_transform(data['NM_PENERIMA'])
    tfidf_matrix_address = vectorizer_address.fit_transform(data['AL_PENERIMA'])
    tfidf_matrix_uraian = vectorizer_uraian.fit_transform(data['URAIAN_BARANG'])
    
    model_ident = NearestNeighbors(metric='cosine', algorithm='brute')
    model_name = NearestNeighbors(metric='cosine', algorithm='brute')
    model_address = NearestNeighbors(metric='cosine', algorithm='brute')
    model_uraian = NearestNeighbors(metric='cosine', algorithm='brute')
    
    model_ident.fit(tfidf_matrix_ident)
    model_name.fit(tfidf_matrix_name)
    model_address.fit(tfidf_matrix_address)
    model_uraian.fit(tfidf_matrix_uraian)
    
    return model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian

# Function to find similar records
def find_similar(ident, name, address, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, top_n=100):
    input_vector_ident = vectorizer_ident.transform([ident])
    input_vector_name = vectorizer_name.transform([name])
    input_vector_address = vectorizer_address.transform([address])
    
    distances_ident, indices_ident = model_ident.kneighbors(input_vector_ident, n_neighbors=top_n)
    distances_name, indices_name = model_name.kneighbors(input_vector_name, n_neighbors=top_n)
    distances_address, indices_address = model_address.kneighbors(input_vector_address, n_neighbors=top_n)

    results = []
    for i in range(top_n):
        index_ident = indices_ident[0][i]
        index_name = indices_name[0][i]
        index_address = indices_address[0][i]
        
        original_ident = df.iloc[index_ident]['NO_IDENT']
        original_name = df.iloc[index_name]['NM_PENERIMA']
        original_address = df.iloc[index_address]['AL_PENERIMA']
        
        levenshtein_distance_ident = Levenshtein.distance(ident, original_ident)
        levenshtein_distance_name = Levenshtein.distance(name, original_name)
        levenshtein_distance_address = Levenshtein.distance(address, original_address)
        
        max_len_ident = max(len(ident), len(original_ident))
        max_len_name = max(len(name), len(original_name))
        max_len_address = max(len(address), len(original_address))
        
        similarity_ident = (max_len_ident - levenshtein_distance_ident) / max_len_ident * 100
        similarity_name = (max_len_name - levenshtein_distance_name) / max_len_name * 100
        similarity_address = (max_len_address - levenshtein_distance_address) / max_len_address * 100
        
        avg_similarity = (similarity_ident * 0.0 + similarity_name * 0.8 + similarity_address * 0.2)
        
        if avg_similarity > 20:
            original_id_aju = df.iloc[index_name]['ID_AJU']  # Changed to index_name
            original_hs_code = df.iloc[index_name]['HS_CODE']  # Changed to index_name
            original_urian_barang = df.iloc[index_name]['URAIAN_BARANG']  # Changed to index_name
            original_jml_sat_hrg = df.iloc[index_name]['JML_SAT_HRG']  # Changed to index_name
            original_cif_detail = df.iloc[index_name]['CIF_DETAIL']  # Changed to index_name
            
            results.append({
                'ID_AJU': original_id_aju,
                'HS_CODE': original_hs_code,
                'URAIAN_BARANG': original_urian_barang,
                'NO_IDENT': original_ident,
                'NM_PENERIMA': original_name,
                'AL_PENERIMA': original_address,
                'JML_SAT_HRG': original_jml_sat_hrg,
                'CIF_DETAIL': original_cif_detail,
                'Similarity (%)': avg_similarity
            })
    return pd.DataFrame(results)

def find_similar_uraian(uraian, df, model_uraian, vectorizer_uraian, top_n=100):
    input_vector_uraian = vectorizer_uraian.transform([uraian])
    
    distances_uraian, indices_uraian = model_uraian.kneighbors(input_vector_uraian, n_neighbors=top_n)

    results = []
    for i in range(top_n):
        index_uraian = indices_uraian[0][i]
        
        original_uraian = df.iloc[index_uraian]['URAIAN_BARANG']
        original_cif = df.iloc[index_uraian]['CIF_DETAIL']
        
        levenshtein_distance_uraian = Levenshtein.distance(uraian, original_uraian)
        
        max_len_uraian = max(len(uraian), len(original_uraian))
        
        similarity_uraian = (max_len_uraian - levenshtein_distance_uraian) / max_len_uraian * 100
        
        avg_similarity = (similarity_uraian * 1.0 )
        
        if avg_similarity > 20:
            original_harga = df.iloc[index_uraian]['JML_SAT_HRG']  # Changed to index_name
            
            results.append({
                'URAIAN_BARANG': original_uraian,
                'JML_SAT_HRG': original_harga,
                'CIF_DETAIL': original_cif,
                'Similarity (%)': avg_similarity
            })
    return pd.DataFrame(results)
  
# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text
  
def get_range(uraian, df, model_uraian, vectorizer_uraian, sentence_model):
    dfx = find_similar_uraian(uraian, df, model_uraian, vectorizer_uraian)
    dfx['SEM_SIMILAR'] = dfx.apply(lambda row: cosine_similarity(sentence_model.encode(clean_text(row['URAIAN_BARANG'])), sentence_model.encode(clean_text(uraian))) * 100, axis=1)
    dfx = dfx[dfx['SEM_SIMILAR'] > 40]
    dfx['HRG_SATUAN'] = dfx.apply(lambda row: float(row['CIF_DETAIL']) / float(row['JML_SAT_HRG']), axis=1)
    # min hrg satuan is q 25%
    min_hrg_satuan = dfx['HRG_SATUAN'].quantile(0.25)
    # max hrg satuan is q 75%
    max_hrg_satuan = dfx['HRG_SATUAN'].quantile(0.75)
    return min_hrg_satuan, max_hrg_satuan
  


