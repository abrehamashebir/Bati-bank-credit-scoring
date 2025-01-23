
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
def load_data(file_path):

    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        logging.error(f'Unable to load Data: {e}')
    
    if df is not None:
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed')]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
    
    return df
    
