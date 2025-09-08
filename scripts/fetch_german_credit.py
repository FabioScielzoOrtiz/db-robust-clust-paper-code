import os
import pandas as pd
from aif360.sklearn.datasets import fetch_german

# Fetch German Credit Data
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, '..', 'data')
filename = 'uci_german_credit.csv'
save_path = os.path.join(data_path, filename)
X, y = fetch_german()
X = X.reset_index(drop=True)
y = pd.DataFrame(y.reset_index(drop=True))
df = pd.concat([y, X], axis=1)
df.to_csv(save_path, index=False)
print(f'German Credit fetched successfully and saved at {save_path}')