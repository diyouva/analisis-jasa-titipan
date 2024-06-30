import streamlit as st
import pandas as pd
import numpy as np
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from sentence_transformers import SentenceTransformer
from hs_risk import get_risk
from price_range import get_range
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Title of the app
st.title('ANALISA JASA TITIPAN')

# A simple text
st.write('Creating a Better CN Through Data')

# Input from user
st.sidebar.title('Similar Importir')
no_ident = st.sidebar.number_input('NO_IDENTITAS', step=1, min_value=0, value=0)
nm_penerima = st.sidebar.text_input('NAMA')
al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
uraian_barang = st.sidebar.text_input('URAIAN BARANG')

# Predict button
if st.sidebar.button('Predict'):
    # Ensure inputs are not empty
    if not nm_penerima or not al_penerima or not uraian_barang:
        st.error('Please provide valid inputs for NAMA, ALAMAT PENERIMA, and URAIAN BARANG')
    else:
        try:
            # Load data in batches
            chunksize = 10000  # Adjust based on memory limits
            df = pd.read_csv('./data/cn1.csv', chunksize=chunksize)
            processed_chunks = []
            for chunk in df:
                chunk = chunk.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
                chunk = chunk[chunk['HS_CODE'].replace('', np.nan).notna()]
                processed_chunks.append(chunk)
            df = pd.concat(processed_chunks)

            logging.info("Data loaded and processed successfully")

            # Mencari Kemiripan Importir
            model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
            sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
            
            similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
            similarity_penerima = similar_id[similar_id['Similarity (%)'] > 60].head(10)  # get similar_id that Similarity (%) > 60%

            logging.info("Similar importers found successfully")

            # Mencari kesesuaian HS Code dengan Nama Produk
            df_hs = pd.read_csv('./data/hs_code_not_clean_id.csv', dtype=str)
            df_hs_results = get_similarity(similar_id, sentence_model, df_hs)

            logging.info("HS Code similarity processed successfully")

            # Mencari range harga berdasarkan uraian produk
            range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
            min_harga = range_harga[0]
            max_harga = range_harga[1]

            logging.info("Price range calculated successfully")

            # Displaying results
            st.title('Displaying a Table in Streamlit')
            st.write('Here is a sample table:')
            st.table(similarity_penerima)

            # Display additional results if needed
            st.write('HS Code Compliance Results:')
            st.table(df_hs_results)

            st.write(f'Price Range for {uraian_barang}:')
            st.write(f'Minimum Price: {min_harga}')
            st.write(f'Maximum Price: {max_harga}')

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"An error occurred: {e}")
