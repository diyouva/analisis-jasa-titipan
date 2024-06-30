import Levenshtein
import numpy as np
import re
import pandas as pd

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
    dfx['HRG_SATUAN'] = dfx.apply(lambda row: float(row['CIF_DETAIL']) / float(row['JML_SAT_HRG']) if float(row['JML_SAT_HRG']) != 0 else float(row['CIF_DETAIL']), axis=1)
    # min hrg satuan is q 25%
    min_hrg_satuan = dfx['HRG_SATUAN'].quantile(0.25)
    # max hrg satuan is q 75%
    max_hrg_satuan = dfx['HRG_SATUAN'].quantile(0.75)
    return min_hrg_satuan, max_hrg_satuan

