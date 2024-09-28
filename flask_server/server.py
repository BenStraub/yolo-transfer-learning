from flask import Flask, request, render_template
#from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the trained model
#model = YOLO('rice_leaf_disease_detection.pt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('data/', file.filename)
        file.save(file_path)

        # Predict using the model
        #results = model.predict(source=file_path, show=True)
        results = "Results" 

        # Process results (just returning the raw results for simplicity)
        return str(results)

if __name__ == '__main__':
    app.run(debug=True)
