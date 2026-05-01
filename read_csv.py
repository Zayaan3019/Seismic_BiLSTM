import pandas as pd

try:
    df = pd.read_csv('NGA_Subduction_filtered.csv')
    print(df.dtypes)
except FileNotFoundError:
    print("Error: The file 'NGA_Subduction_filtered.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")