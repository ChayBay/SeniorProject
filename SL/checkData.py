import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

trainData = np.load('finalData.npy',allow_pickle=True)
df = pd.DataFrame(trainData)


print(df)

none = [0, 0, 0, 0, 0, 0, 0, 0]
index = 0
for i in df.values:
    if i[1] == none:
        print("jere")
        df.drop(index, inplace = True)
    index+=1
    
print(df)

