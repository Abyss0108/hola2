from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        tenure = float(request.form['tenure'])
        TotalCharges = float(request.form['TotalCharges'])
        PaymentMethod = float(request.form['PaymentMethod'])
        PhoneService = float(request.form['PhoneService'])
        PaperlessBilling = float(request.form['PaperlessBilling'])
        MonthlyCharges = float(request.form['MonthlyCharges'])
        Contract = float(request.form['Contract'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[tenure, TotalCharges, PaymentMethod, PhoneService, PaperlessBilling, MonthlyCharges, Contract]], 
                               columns=['tenure', 'TotalCharges', 'PaymentMethod', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'Contract'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)