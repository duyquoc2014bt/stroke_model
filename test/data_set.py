import pandas as pd

data = pd.read_excel("test/data.xlsx", engine="openpyxl", header=0)

print(data, "dq")