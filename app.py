import joblib 
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd

app=Flask(__name__)
model=joblib.load(open('aqi_prediction_pipeline.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    df=pd.DataFrame([data])
    print(df)
    output=model.predict(df)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)    