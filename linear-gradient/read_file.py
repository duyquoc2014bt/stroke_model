import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "data.xlsx")

data = pd.read_excel(file_path, engine="openpyxl", header=2)