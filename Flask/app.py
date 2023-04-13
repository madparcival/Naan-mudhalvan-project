from flask import Flask,request,render_template 
import numpy as np
import pandas as pd
import pickle 
import os
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def prediction():
    flnum = int(request.form['flightnumber'])
    month = int(request.form['month'])
    day   = int(request.form['day'])
    week   = int(request.form['week'])
    schdl_dep = float(request.form['schdl_dep'])
    dep_delay = int(request.form['dep_delay'])
    schdl_arriv = int(request.form['schdl_arriv'])
    divrtd = int(request.form['divrtd'])
    cancld = int(request.form['cancld'])
    air_sys_delay = float(request.form['air_sys_delay'])
    secrty_delay = float(request.form['secrty_delay'])
    airline_delay = float(request.form['airline_delay'])
    late_air_delay = float(request.form['late_air_delay'])
    wethr_delay  = float(request.form['wethr_delay'])
    
    inputvariables=[[flnum,month,day,week,schdl_arriv,dep_delay]]
    np.array(inputvariables)
    model = pickle.load(open('flightDTCmodel.pkl','rb'))
    
    pred=model.predict(sc.transform(inputvariables))
        
    return render_template("result.html",prediction=pred)

if __name__=='__main__':
    app.run(debug=True)
