import pandas as pd


def load_csv(path: str):
    try:
        content = pd.read_csv(path)
        return content
    except Exception as e:
        print(e)
        return None
