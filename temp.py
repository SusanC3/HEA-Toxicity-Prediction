import pandas as pd
import pdb

df = pd.read_csv("assays/AID_1671196_p450-2d6-p1.csv", skiprows=[i for i in range(50, 9667)])
df = df.drop([df.columns[i] for i in range(100, 2049)], axis=1)

#pdb.set_trace()

df.to_csv("shortened_p450.csv")