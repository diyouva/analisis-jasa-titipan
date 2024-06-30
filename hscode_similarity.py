import re
import numpy as np

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_similarity(df, model, df_hs):
    dfx = df.copy()
    # dfx POS_DESC get 6 digit from HS_CODE then get the description from df_hs
    dfx['POS'] = dfx['HS_CODE'].str[:6]
    # using data in POS, find description in df_hs['DESCRIPTION'] and put it in dfx['POS_DESC']
    dfx['POS_DESC'] = dfx['POS'].apply(lambda x: df_hs[df_hs['POS'] == x]['DESCRIPTION'].values[0] if df_hs[df_hs['POS'] == x]['DESCRIPTION'].values.size else np.nan)
    dfx.drop(columns=['POS'], inplace=True)
    dfx = dfx[['ID_AJU', 'HS_CODE','URAIAN_BARANG','POS_DESC', 'JML_SAT_HRG', 'CIF_DETAIL']].copy()
    # remove row that hscode is empty string or nan
    dfx = dfx[dfx['POS_DESC'].notna()]
    dfx['SIMILARITY'] = dfx.apply(lambda row: cosine_similarity(model.encode(clean_text(row['URAIAN_BARANG'])), model.encode(clean_text(row['POS_DESC']))), axis=1)
    return dfx