from flask import Flask,request,render_template 
import pickle 
from sklearn.preprocessing import StandardScaler

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
    dep_delay = float(request.form['dep_delay'])
    schdl_arriv = int(request.form['schdl_arriv'])
    
    inputvariables=[[flnum,month,day,week,schdl_arriv,dep_delay]]
    model = pickle.load(open('flightRFCmodel.pkl','rb'))
    
    pred=model.predict(inputvariables)
    
    if pred == 1:
        pred='Will be'
    else:
        pred="Won't get"

    return render_template("result.html",flightnumber=flnum,prediction=pred)

if __name__=='__main__':
    app.run(debug=True)
