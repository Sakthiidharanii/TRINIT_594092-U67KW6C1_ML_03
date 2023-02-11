import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import requests, json
import warnings
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/final', methods=['POST', 'GET'])

def predict():  
    warnings.filterwarnings('ignore')

    soil_type = str(request.form.get('soiltype'))
    districtt = request.form.get('district')
    
    d=pd.read_csv('rainfall in india 1901-2015.csv')
    d=d.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    d=d.get_group(('TAMIL NADU'))
    df=d.melt(['YEAR']).reset_index()
    df= df[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
    df.columns=['Index','Year','Month','Avg_Rainfall']
    Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    df['Month']=df['Month'].map(Month_map)
    X=np.asanyarray(df[['Year','Month']]).astype('int')
    y=np.asanyarray(df['Avg_Rainfall']).astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_test_predict=LR.predict(X_test)
    data = np.array([[2023,6]])
    rain=LR.predict(data)

    api_key = "66e08bc428195d5002615ea10b4b1ef0"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + districtt
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temp = y["temp"]
        humid = y["humidity"]

    d=pd.read_csv('soil dataset.csv')
    d.columns=['soiltype','Ni','Po','Ko','PH']
    for index,row in d.iterrows():
        if row['soiltype'] ==soil_type:
            s=row
            


    df=pd.read_csv('Crop_recommendation.csv')
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    NaiveBayes = GaussianNB()
    NaiveBayes.fit(Xtrain,Ytrain)
    data = np.array([[s['Ni'],s['Po'], s['Ko'], temp- 273.15,humid, s['PH'], rain]])
    prediction = NaiveBayes.predict(data)
    

    if prediction == "coconut":
        cr = 2
    elif prediction == "tobacco":
        cr = 23
    elif prediction == "turmeric":
        cr = 24
    elif prediction == "copra":
        cr = 4
    elif prediction == "maize":
        cr = 15
    elif prediction == "paddy":
        cr = 16
    elif prediction == "groundnut":
        cr = 9
    elif prediction == "sesamum":
        cr = 18
    elif prediction == "t.v.cumbu":
        cr = 19
    elif prediction == "hybrid cumbu":
        cr = 10
    elif prediction == "ragi":
        cr = 17
    elif prediction == "thinai":
        cr = 22
    elif prediction == "black gram":
        cr = 0
    elif prediction == "green gram":
        cr = 8
    elif prediction == "tamirind fruit":
        cr = 20
    elif prediction == "coconut seed":
        cr = 3
    elif prediction == "cotton":
        cr = 5
    elif prediction == "Cashewnuts":
        cr = 1
    elif prediction == "Tapioca":
        cr = 21
    elif prediction == "Cowpea":
        cr = 6
    elif prediction == "Kulthi":
        cr = 14
    elif prediction == "Ragi":
        cr = 17
    elif prediction == "Gingelly Oil":
        cr = 7
    elif prediction == "millet":
        cr = 13
    else:
        cr = -1

    if districtt == "Coimbatore":
        districtt= 0
    elif districtt == "Cuddalore":
        districtt= 1
    elif districtt == "Dharmapuri":
        districtt= 2
    elif districtt == "Dindigul":
        districtt= 3
    elif districtt == "Erode":
        districtt= 4
    elif districtt == "Kancheepuram":
        districtt= 5
    elif districtt == "Krishnagiri":
        districtt= 6
    elif districtt == "Madurai":
        districtt= 7
    elif districtt == "Nagercoil":
        districtt= 8
    elif districtt == "Namakkal":
        districtt= 9
    elif districtt == "Ramanathapuram":
        districtt= 10
    elif districtt == "Salem":
        districtt= 11
    elif districtt == "Sivagangai":
        districtt= 12
    elif districtt == "Thanjavur":
        districtt= 13
    elif districtt == "Thiruvannamalai":
        districtt= 14
    elif districtt == "Vellore":
        districtt= 15
    elif districtt == "Villupuram":
        districtt= 16
    elif districtt == "Virudhunagar":
        districtt = 17

    d=pd.read_csv('crop.csv')
    df = d[d['state'] == 'Tamil Nadu']
    encoder = LabelEncoder()
    df['district'] = encoder.fit_transform(df['district'])
    df['commodity'] = encoder.fit_transform(df['commodity'])
    X=df.iloc[:,[1,3]]
    y=df.iloc[:,6:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    LR = LinearRegression()
    LR.fit(X_train,y_train)
    y_test_predict=LR.predict(X_test)
    if cr!=-1:
        data = np.array([[districtt,cr]])
        price=LR.predict(data)
        return render_template('final.html', crop = prediction, cost =price[2] )
    else:
        return render_template('final.html',crop = prediction, cost = "Price detail temporarily not found")
if __name__ == '__main__':
    app.run(debug=True)