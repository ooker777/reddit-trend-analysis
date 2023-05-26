import pandas as pd
import os


print(os.getcwd())
path = os.getcwd()+"/NLP-Reddit/data/posts5000.pkl"
print(path)
df = pd.read_pickle(path)
df.info(memory_usage='Deep')