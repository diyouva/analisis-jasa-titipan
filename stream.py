import streamlit as st
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from hs_risk import get_risk
from price_range import get_range
import numpy as np

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
        # Load data
        df = load_data('./data/cn1.csv')
        df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
        df = df[df['HS_CODE'].replace('', np.nan).notna()]

        # Mencari Kemiripan Importir
        model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
        sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        try:
            similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
            similarity_penerima = similar_id[similar_id['Similarity (%)'] > 60].head(10)  # get similar_id that Similarity (%) > 60%
        except Exception as e:
            st.error(f"Error finding similar importers: {e}")
            st.stop()

        # Mencari kesesuaian HS Code dengan Nama Produk
        try:
            df_hs = pd.read_csv('./data/hs_code_not_clean_id.csv', dtype=str)
            df_hs_results = get_similarity(similar_id, sentence_model, df_hs)
        except Exception as e:
            st.error(f"Error processing HS code similarity: {e}")
            st.stop()

        # Mencari range harga berdasarkan uraian produk
        try:
            range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
            min_harga = range_harga[0]
            max_harga = range_harga[1]
        except Exception as e:
            st.error(f"Error calculating price range: {e}")
            st.stop()

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