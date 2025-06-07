def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df