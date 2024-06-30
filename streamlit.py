# import function create index from file 1_similar.py
import streamlit as st
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from hs_risk import get_risk
from price_range import get_range
import numpy as np

def main ():
    # Title of the app
    st.title('ANALISA JASA TITIPAN')
    
    # A simple text
    st.write('Creating a Better CN Through Data')
    
# Input from user
st.sidebar.title('Similar Importir')
no_ident = st.sidebar.number_input ('NO_IDENTITAS')
nm_penerima = st.sidebar.text_input('NAMA')
al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
uraian_barang = st.sidebar.text_input('URAIAN BARANG')

#Predict button
if st.sidebar.button('Predict'):
    model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
    sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    X = np.array([no_ident, nm_penerima, al_penerima, uraian_barang])
    if any(len(x.strip()) == 0 for x in X):
      st.markdown('### All inputs must be non-empty and non-whitespace')
    else:
      try:
          # Ensure the input is in the right shape for the model
          input_data = [X]

          # Call the find_similar function
          similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)

          # Filter the DataFrame
          filtered_df = similar_id[similar_id['Similarity (%)'] > 60].head(10)

          # Display the filtered DataFrame
          st.markdown(f'### Filtered Similarity Results:')
          st.write(filtered_df)
      except Exception as e:
          st.markdown('### An error occurred during model prediction')
          st.write(str(e))

# #Price Range
# st.sidebar.title('Price Range by Description')
# uraian_barang = st.text_input('Uraian Barang')

# if st.sidebar.button('Predict'):
#     range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
#     # print range harga min and max, tuple (min, max)
#     print(f'Harga kisaran min: {range_harga[0]}')
#     print(f'Harga kisaran max: {range_harga[1]}')
        
if __name__ == '__main__':
    main()


