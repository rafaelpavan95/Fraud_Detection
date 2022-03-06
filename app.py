import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np

class Details(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float 

app = FastAPI()

with open('xgboost.pkl', 'rb') as m:
    model = joblib.load(m)


@app.get('/')
def index():
    return {'message': "Rafael Pavan API."}


@app.get('/{name}')
def get_name(name: str):

    return {'message': "This is the home page of this API. Go to /prediction to use the Machine Learning model."}


@app.get('/{name}')
def index(name: str):

    return {'message': "This is the home page of this API. Go to /prediction to use the Machine Learning model."}

@app.post('/prediction')

def classify_transaction(dados: Details):

    received = dados.dict()
    Time =   received['Time']
    V1 = received['V1']
    V2 = received['V2']
    V3 = received['V3']
    V4 = received['V4']
    V5 = received['V5']
    V6 = received['V6']
    V7 = received['V7']
    V8 = received['V8']
    V9 = received['V9']
    V10 = received['V10']
    V11 = received['V11']
    V12 = received['V12']
    V13 = received['V13']
    V14 = received['V14']
    V15 = received['V15']
    V16 = received['V16']
    V17 = received['V17']
    V18 = received['V18']
    V19 = received['V19']
    V20 = received['V20']
    V21 = received['V21']
    V22 = received['V22']
    V23 = received['V23']
    V24 = received['V24']
    V25 = received['V25']
    V26 = received['V26']
    V27 = received['V27']
    V28 = received['V28']
    Amount =  received['Amount'] 

    classified_transaction = model.predict(np.array([[Time, V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount]])).tolist()[0]
    
    if classified_transaction==1:

        var = 'Fraudulent Transaction'

    else: var = 'Genuine Transaction'
    
    return {'message': var, 'label':classified_transaction}



if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
    
