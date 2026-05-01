import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

try:
    # CSV file path
    csv_file = 'NGA_Subduction_filtered.csv'

    # Input columns
    input_columns = [
        'Earthquake_Magnitude',
        'Ztor_km',
        'Finite_Fault_Model',
        'Rjb_km',
        'Vs30_Selected_for_Analysis_m_s',
    ]

    # Output columns
    output_columns = [
        'PGA_g',
        'T0pt010S', 'T0pt020S', 'T0pt030S', 'T0pt050S',
        'T0pt050S', 'T0pt060S', 'T0pt070S', 'T0pt080S',
        'T0pt090S', 'T0pt100S', 'T0pt200S', 'T0pt300S',
        'T0pt400S', 'T0pt500S', 'T0pt600S', 'T0pt700S',
        'T0pt800S', 'T0pt900S', 'T1pt000S', 'T2pt000S',
        'T3pt000S', 'T4pt000S', 'T5pt000S', 'T6pt000S',
        'T7pt000S', 'T8pt000S', 'T9pt000S', 'T10pt000S'
    ]

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    original_indices = df.index

    # Select the input columns
    input_data = df[input_columns].copy()

    # Convert input columns to numeric, coercing errors to NaN
    for col in input_columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    # Fill NaN values with 0
    input_data = input_data.fillna(0)

    # Calculate log(Rjb_km) and add it to input_data
    input_data['log(Rjb_km)'] = np.log(input_data['Rjb_km'])
    input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_data = input_data.fillna(0)

    # Select the output columns
    output_data = df[output_columns].copy()
    print(output_data.columns)

    # Scale the input data
    scaler = StandardScaler()
    scaled_input_data = scaler.fit_transform(input_data)

    # Scale the output data
    output_scaler = StandardScaler()
    scaled_output_data = output_scaler.fit_transform(output_data)

    # Define the model
    model = Sequential()
    model.add(Dense(128, input_dim=scaled_input_data.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(scaled_output_data.shape[1]))  # Output layer with the number of output features

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(scaled_input_data, scaled_output_data, epochs=10, batch_size=32, verbose=0)

    # Make predictions
    scaled_predictions = model.predict(scaled_input_data)

    # Inverse transform the scaled predictions
    predictions = output_scaler.inverse_transform(scaled_predictions)
    predictions_df = pd.DataFrame(predictions, index=original_indices)
    max_val = 1.1 * np.max([predictions,output_data.values])

    # Plotting for each output column
    for i, col in enumerate(output_columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(output_data[col], predictions_df.loc[:, col], alpha=0.5)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted - {col}")
        plt.xlim(0,max_val)
        plt.ylim(0,max_val)
        plt.savefig(f'{col}.png')
        plt.close()

    print("Plots generated successfully!")

except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")