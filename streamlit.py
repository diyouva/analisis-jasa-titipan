# import function create index from file 1_similar.py
import streamlit as st
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from hs_risk import get_risk
from price_range import get_range
import numpy as np

# def main ():
    # Title of the app
st.title('ANALISA JASA TITIPAN')

# A simple text
st.write('Creating a Better CN Through Data')

# Load data
df = load_data('./data/cn1.csv')
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df[df['HS_CODE'].replace('', np.nan).notna()]

# Input from user
st.sidebar.title('Similar Importir')
no_ident = st.sidebar.text_input('NO_IDENTITAS')
nm_penerima = st.sidebar.text_input('NAMA')
al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
uraian_barang = st.sidebar.text_input('URAIAN BARANG')

# Function
## Mencari Kemiripan Importir
model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
similarity_penerima = similar_id[similar_id['Similarity (%)'] > 60].head(10) # get similar_id that Similarity (%) > 0.6   

## Mencari kesesuaian HS Code dengan Nama Produk
df_hs = pd.read_csv('./data/hs_code_not_clean_id.csv', dtype=str)
df_hs_results = get_similarity(similar_id, sentence_model, df_hs)

## Mencari range harga berdasarkan uraian produk
range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
min_harga = range_harga[0]
max_harga = range_harga[1]

#Predict button
if st.sidebar.button('Predict'):
    # Displaying results
    st.title('Displaying a Table in Streamlit')
    st.write('Here is a sample table:')
    st.table(similarity_penerima)    # Mencari kesesuaian HS Code dengan Nama Produk
    
    # Mencari range harga berdasarkan uraian produk



    # X = np.array([no_ident, nm_penerima, al_penerima, uraian_barang])
    # if any(len(x.strip()) == 0 for x in X):
    #   st.markdown('### All inputs must be non-empty and non-whitespace')
    # else:
    #   try:
    #       # Ensure the input is in the right shape for the model
    #       input_data = [X]

    #       # Call the find_similar function
    #       similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)

    #       # Filter the DataFrame
    #       filtered_df = similar_id[similar_id['Similarity (%)'] > 60].head(10)

    #       # Display the filtered DataFrame
    #       st.markdown(f'### Filtered Similarity Results:')
    #       st.write(filtered_df)
    #   except Exception as e:
    #       st.markdown('### An error occurred during model prediction')
    #       st.write(str(e))

# #Price Range
# st.sidebar.title('Price Range by Description')
# uraian_barang = st.text_input('Uraian Barang')

# if st.sidebar.button('Predict'):
#     range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
#     # print range harga min and max, tuple (min, max)
#     print(f'Harga kisaran min: {range_harga[0]}')
#     print(f'Harga kisaran max: {range_harga[1]}')
        
# if __name__ == '__main__':
#     main()


