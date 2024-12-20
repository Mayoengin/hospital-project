import pandas as pd
import joblib
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, request
    
app = Flask(__name__)

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

arima_model_t2 = joblib.load('arima_model_t2.pkl')
arima_model_t3 = joblib.load('arima_model_t3.pkl')
arima_model_t4 = joblib.load('arima_model_t4.pkl')

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        excel_file_path = os.path.join(bundle_dir, 'over.xlsx')

        df = pd.read_excel(excel_file_path)

        t1 = df[['Gewicht_T1', 'T1', 'FFM%_T1', 'FM%_T1', 'Pha_T1', 'RZ_T1', 'Lengte_T1', 'Age_T1', 'SMI_T1', 'REE_T1', 'Geslacht_T1']]
        t2 = df[['Gewicht_T2', 'T2', 'FFM%_T2', 'FM%_T2', 'Pha_T2', 'RZ_T2', 'Lengte_T2', 'Age_T2', 'SMI_T2', 'REE_T2', 'Geslacht_T2']]
        t3 = df[['Gewicht_T3', 'T3', 'FFM%_T3', 'FM%_T3', 'Pha_T3', 'RZ_T3', 'Lengte_T3', 'Age_T3', 'SMI_T3', 'REE_T3', 'Geslacht_T3']]
        t4 = df[['Gewicht_T4', 'T4', 'FFM%_T4', 'FM%_T4', 'Pha_T4', 'RZ_T4', 'Lengte_T4', 'Age_T4', 'SMI_T4', 'REE_T4', 'Geslacht_T4']]

        arima_option = int(request.form['arima_option'])
        if arima_option == 1:
            arima_model = arima_model_t2
            test_features = pd.concat([t1, t2['T2']], axis=1)
            test_target = df['Gewicht_T2']
        elif arima_option == 2:
            arima_model = arima_model_t3
            test_features = pd.concat([t1, t2, t3['T3']], axis=1)
            test_target = df['Gewicht_T3']
        elif arima_option == 3:
            arima_model = arima_model_t4
            test_features = pd.concat([t1, t2, t3, t4['T4']], axis=1)
            test_target = df['Gewicht_T4']
        else:
            return "Invalid option."

        scaler = StandardScaler()
        new_features_scaled = scaler.fit_transform(test_features)

        # Predict the entire sequence using the ARIMA model
        predicted_mean = arima_model.get_forecast(steps=len(test_target), exog=new_features_scaled).predicted_mean
        # Initialize an empty list to store patient IDs and predictions
        patient_ids = []
        predictions = []

        # Loop through the patient IDs and predictions
        for patient_id, prediction in zip(df['Patient_ID'], predicted_mean):
            patient_ids.append(patient_id)
            predictions.append(prediction)

        # Create a DataFrame with 'Patient_ID' and 'Predicted_Gewicht' columns
        predictions_df = pd.DataFrame({'Patient_ID': patient_ids, 'Predicted_Gewicht': predictions})
        
        target_id = int(request.form['Patient_ID'])
            
        # Filter the predictions DataFrame to get predictions for the specified patient ID
        selected_patient_prediction = predictions_df[predictions_df['Patient_ID'] == target_id]

        # Check if the selected_patient_prediction DataFrame is empty
        if selected_patient_prediction.empty:
            # Render the result.html template with an empty 'predictions' variable
            return render_template('result.html', predictions=[])
        else:
            # Render the result.html template with the selected patient's prediction
            return render_template('result.html', predictions=selected_patient_prediction.to_dict('records'))

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
