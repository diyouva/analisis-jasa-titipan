# import function create index from file 1_similar.py
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from hs_risk import get_risk
from price_range import get_range
import numpy as np

sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

no_ident = ''
nm_penerima = 'suwanto'
al_penerima = 'jl. raya bogor km 30'
uraian_barang = 'beras'


# Load and preprocess the data
filepath_cn = './data/cn1.csv'
def load_data(filepath_cn):
    df = load_data(filepath_cn)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df['HS_CODE'].replace('', np.nan).notna()]
    return df

# Pencarian importir / penerima
def search_importir(df):
    model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
    similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
    similarity_penerima = similar_id[similar_id['Similarity (%)'] > 60].head(10) # get similar_id that Similarity (%) > 0.6
    return similar_id, similarity_penerima, model_uraian, vectorizer_uraian

# Mencari kesesuaian HS Code dengan Nama Produk
filepath_produk = './data/hs_code_not_clean_id.csv'
def load_data(filepath_produk, similar_id):
    df_hs = pd.read_csv(filepath_produk, dtype=str)
    df_hs_results = get_similarity(similar_id, sentence_model, df_hs)
    return df_hs_results

# Mencari range harga berdasarkan uraian produk
def range_harga(uraian_barang, df, model_uraian, vectorizer_uraian):
    range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
    min_harga = range_harga[0]
    max_harga = range_harga[1]
    return min_harga, max_harga
