import pandas as pd

def load_client_data(path="data/in/train_clientes_sample.csv"):
    return pd.read_csv(path)

def load_reqs_data(path="data/in/train_requerimientos_sample.csv"):
    return pd.read_csv(path)