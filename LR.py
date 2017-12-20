import pandas as pd
import numpy as np
import io

train_data = pd.read_csv('data/train.csv')
A_dict = {}

for i in range(41):
    count = train_data.iloc[:, i].unique().size
    A_dict[i] = count

print(A_dict)

count_frame = pd.DataFrame.from_dict(A_dict,orient='index').T
count_frame.to_csv('temp.csv', index=False)