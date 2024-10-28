import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import custom functions from your modules
from create_index import create_index, load_data, find_similar
from hscode_similarity import get_similarity
from price_range import get_range

# Load data
df = load_data('./data/cn1.csv')
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df[df['HS_CODE'].replace('', np.nan).notna()]
# Check if 'HS_CODE' column exists
if 'HS_CODE' not in df.columns:
    st.error("The 'HS_CODE' column is missing from the data.")

df_hs = pd.read_csv('./data/hs_code_not_clean_id.csv', dtype=str)
# Check if 'HS_CODE' column exists in df_hs
# if 'POS' not in df_hs.columns:
#     st.error("The 'HS_CODE' column is missing from the HS code data.")


st.sidebar.title('Input')
no_ident = st.sidebar.text_input ('NO_IDENTITAS')
nm_penerima = st.sidebar.text_input('NAMA')
al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
uraian_barang = st.sidebar.text_input('URAIAN BARANG')

# Global variables for models
model_ident = None
vectorizer_ident = None
model_name = None
vectorizer_name = None
model_address = None
vectorizer_address = None
model_uraian = None
vectorizer_uraian = None
sentence_model = None


# Define function to initialize models and load data
def initialize():
    global model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian, sentence_model
    model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address, model_uraian, vectorizer_uraian = create_index(df)
    sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)

# Function for Similar Importer Analysis page
def page_about():
    st.title('About')
    st.markdown("""
    This app is designed to analyze and improve the Customs Declaration (CN) process 
    by leveraging data analytics and machine learning models.
    
    ### Features:
    - **Similar Importer Analysis:** Analyzes similarity between importers based on identification, name, and address.
    - **HS Code Search:** Helps in finding HS codes based on product descriptions.
    - **Price Range Prediction:** Estimates price range based on product description.""")
    # Image to explain the goal of the app (replace with your image file path)
    st.image(r"C:\Users\Lenovo\OneDrive\Training Data Scientist\analisis-jasa-titipan-main\analisis-jasa-titipan-main\satu.jpg", caption="Jumlah Dokumen Barang Kiriman", use_column_width=True)
    st.image(r"C:\Users\Lenovo\OneDrive\Training Data Scientist\analisis-jasa-titipan-main\analisis-jasa-titipan-main\dua.jpg", caption="Facts", use_column_width=True)

    st.sidebar.markdown("""
     How to Use:
    1. Enter importer details and product description on the left sidebar.
    2. Click on "Predict" to analyze similar importers or predict price range.
    3. Use tabs to navigate between different functionalities.
    
    For any issues or questions, please contact the app administrator.
    """)

def page_overview_cn():
    st.title('Overview Barang Kiriman')
 # Embedding Tableau dashboard using HTML and JavaScript
    tableau_embed_code = """
    <div class='tableauPlaceholder' id='viz1719808441976' style='position: relative'>
        <noscript><a href='#'><img alt='Overview ' src='https://public.tableau.com/static/images/Da/Dashboard-BarangKiriman/Overview/1_rss.png' style='border: none' /></a></noscript>
        <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='Dashboard-BarangKiriman/Overview' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/Da/Dashboard-BarangKiriman/Overview/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
        </object>
    </div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1719808441976');
        var vizElement = divElement.getElementsByTagName('object')[0];
        vizElement.style.width='1225px';
        vizElement.style.height='783px';
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """

    # Displaying the Tableau embed code in Streamlit
    st.components.v1.html(tableau_embed_code, width=1280, height=830)

def page_similar_importer():
    st.title('Similar Importer Analysis')
    # st.sidebar.title('Inputs')
    # no_ident = st.sidebar.text_input('NO_IDENTITAS')
    # nm_penerima = st.sidebar.text_input('NAMA')
    # al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
    # uraian_barang = st.sidebar.text_input('URAIAN BARANG')

    if st.sidebar.button('Predict'):
        initialize()
        try:
            # similar_id
            similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
            filtered_df = similar_id[similar_id['Similarity (%)'] > 60].head(10)
            st.markdown('### Filtered Similarity Results:')
            st.write(filtered_df)
        except Exception as e:
            st.markdown('### An error occurred during model prediction')
            st.write(str(e))

# Function for Price Range Prediction page
# def page_price_range():
#     st.title('Price Range Prediction')
#     st.sidebar.title('Inputs')
#     # uraian_barang = st.sidebar.text_input('URAIAN BARANG')

#     if st.sidebar.button('Predict Price Range'):
#         initialize()
#         try:
#             range_harga = get_range(uraian_barang, df, model_uraian, vectorizer_uraian, sentence_model)
#             st.write(f'Harga kisaran min: {range_harga[0]}')
#             st.write(f'Harga kisaran max: {range_harga[1]}')
#         except Exception as e:
#             st.write(f'Error predicting price range: {str(e)}')

# Function for HSCode Search page
def page_hscode_search():
    st.title('HS Code Search')
    st.sidebar.title('Inputs')
    # df_hs = df_hs
    # no_ident = st.sidebar.text_input('NO_IDENTITAS')
    # nm_penerima = st.sidebar.text_input('NAMA')
    # al_penerima = st.sidebar.text_input('ALAMAT PENERIMA')
    # uraian_barang = st.sidebar.text_input('URAIAN BARANG')        
    if st.sidebar.button('Search HS Code'):
        
        # if sentence_model is None:
        initialize()  # Ensure models are initialized
        try:
            similar_id = find_similar(no_ident, nm_penerima, al_penerima, df, model_ident, vectorizer_ident, model_name, vectorizer_name, model_address, vectorizer_address)
            df_hs_results = get_similarity(similar_id, sentence_model, df_hs)
            st.markdown('### HS Code Search Results:')
            st.write(df_hs_results)
        except Exception as e:
            st.error(f'Error searching for HS Code: {str(e)}')

# Main function to control page navigation
def main():
    st.image(r"C:\Users\Lenovo\OneDrive\Training Data Scientist\analisis-jasa-titipan-main\analisis-jasa-titipan-main\belanja3.jpg", use_column_width=True)
    st.sidebar.title('Navigation')
    pages = {
        "About" : page_about,
        "Overview Barang Kiriman" : page_overview_cn,
        "Similar Importer Analysis": page_similar_importer,
        # "Price Range Prediction": page_price_range,
        "HS Code Search": page_hscode_search,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page = pages[selection]
    page()

if __name__ == '__main__':
    main()


