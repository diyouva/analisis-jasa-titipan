import pandas as pd
from sklearn.metrics import jaccard_score
from difflib import SequenceMatcher

def load_data(filepath):
    df = pd.read_csv(filepath, dtype=str)
    # df remove all rows with NaN values
    df = df.dropna()
    # in all column, if value contains comma, change it to dot
    df = df.apply(lambda x: x.str.replace(',', '.'))
    # trim last spaces in ppnbm column
    df['ppnbm'] = df['ppnbm'].str.strip()
    # if ppnbm is not '0', get last 2 digits
    df['ppnbm'] = df['ppnbm'].apply(lambda x: x[-2:] if x != '0' else x)
    # remove all rows that has 'kg' or 'liter' in bm column
    df = df[~df['bm'].str.contains('kg|liter|menit')]
    # remove all spaces in all column
    df = df.apply(lambda x: x.str.replace(' ', ''))
    # if column bk contains '', replace it with '0'
    df['bk'] = df['bk'].replace('', '0')
    # convert all column except hs to float
    for col in df.columns[1:]:
        if col != 'hs':
            df[col] = df[col].astype(float)
    # replace all dot in hs column
    df['hs'] = df['hs'].str.replace('.', '')
    return df

# Function to create an index for efficient searching
def create_index(df):
    index = {}
    for i, row in df.iterrows():
        hscode = row['hs']
        index[hscode] = i
    return index

# Function to calculate differences between two HS codes
def calculate_differences(hscode1, hscode2, df, index):
    differences = {}
    for col in ['bm', 'bk', 'ppn', 'ppnbm']:
        differences[col] = df.loc[index[hscode1], col] - df.loc[index[hscode2], col]
    return differences

# Function to find similar HS codes

def test_risk(hscode, index, df):
    # Extract the first 4 digits of the input HS code
    hscode_prefix = hscode[:4]
    # Find all HS codes with the same first 4 digits
    matching_hs_codes = [code for code in index if code.startswith(hscode_prefix)]
    results = []
    for hs in matching_hs_codes:
        # Compare the remaining 4 digits of the HS codes using a similarity metric
        hscode_suffix = hscode[4:]
        hs_suffix = hs[4:]
        similarity_ratio = SequenceMatcher(None, hscode_suffix, hs_suffix).ratio()
        similarity_percent = int(similarity_ratio * 100)
        # You can adjust the similarity threshold as needed
        if similarity_percent >= 75:
            # Example: Require at least 75% similarity
            differences = calculate_differences(hscode, hs, df, index)
            result = {'HSCode': hs, 'similarity': similarity_percent, **differences}
            results.append(result)
    df_results = pd.DataFrame(results)
    # Check if the required columns exist in df_results
    required_columns = ['bm', 'bk', 'ppn', 'ppnbm']
    if all(col in df_results.columns for col in required_columns):
        total_sum = df_results[required_columns].sum()
        if (total_sum < 0).any():
            risk = 'Y'
        else:
            risk = 'N'
    else:
        # Handle the case where the required columns are missing
        risk = 'N'
    return risk

  
def get_risk(df):
    df_btki = load_data('btki2.csv')
    index = create_index(df_btki)
    # dfx['FL_RISK2'] = dfx['HS_CODE'].apply(lambda x: test_risk(x))
    df['FL_RISK2'] = df['HS_CODE'].apply(lambda x: test_risk(x, index, df_btki))
    
    return df