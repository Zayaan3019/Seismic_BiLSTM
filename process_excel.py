import pandas as pd

try:
    # Define input and output columns
    input_columns = ['Earthquake_Magnitude', 'Ztor_km', 'Finite_Fault_Model', 'Rjb_km', 'log(Vs30_Selected_for_Analysis_m_s)', 'log(Rjb_km)']
    output_columns = ['PGA_g', 'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt050S', 'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S', 'T0pt090S', 'T0pt100S', 'T0pt200S', 'T0pt300S', 'T0pt400S', 'T0pt500S', 'T0pt600S', 'T0pt700S', 'T0pt800S', 'T0pt900S', 'T1pt000S', 'T2pt000S', 'T3pt000S', 'T4pt000S', 'T5pt000S', 'T6pt000S', 'T7pt000S', 'T8pt000S', 'T9pt000S', 'T10pt000S']

    # Read the Excel file
    df = pd.read_excel('NGA_Subduction_filtered.xlsx')

    # Select the input and output columns
    input_df = df[input_columns]
    output_df = df[output_columns]

    # Print the first 5 rows of the selected input and output columns
    print("First 5 rows of input columns:")
    print(input_df.head())
    print("\nFirst 5 rows of output columns:")
    print(output_df.head())

except FileNotFoundError:
    print("Error: The file 'NGA_Subduction_filtered.xlsx' was not found.")
except KeyError as e:
    print(f"Error: Column not found in the Excel file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")