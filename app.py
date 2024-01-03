from flask import Flask,render_template,request
from args import location_mapping,status_mapping,direction_mapping,property_type_mapping
import numpy as np
import pickle
with open('Model.pkl','rb') as mod:
    model=pickle.load(mod)
with open('Scaler.pkl','rb') as mod:
    scaler=pickle.load(mod)    
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def index():
    #print(request.method)
    #print(request.form)
    if request.method=='POST':
        bedrooms=request.form['bedrooms']
        bathrooms=request.form['bathrooms']
        location=request.form['location']
        sft=request.form['sft']
        status=request.form['status']
        direction=request.form['direction']
        propert_ytype=request.form['property type']
        input_array=np.array([[bedrooms,
                                bathrooms,
                                location,
                                sft,
                                status,
                                direction,
                                propert_ytype]])
        t_array=scaler.transform(input_array) 
        prediction=model.predict(t_array)[0]
        return render_template('index.html',location_mapping=location_mapping,
                         status_mapping=status_mapping,
                         direction_mapping=direction_mapping,
                         property_type_mapping=property_type_mapping , prediction=prediction)   
    else:
        return render_template('index.html',location_mapping=location_mapping,
                         status_mapping=status_mapping,
                         direction_mapping=direction_mapping,
                         property_type_mapping=property_type_mapping   )     
    #return 'I am in first page'
    #return render_template('index.html')
@app.route('/second')
def second():
    return 'I am in second page'
@app.route('/thrid')
def thrid():
    return 'I am in thrid page'
@app.route('/four')
def four():
    return 'I am in four page'
if __name__=='__main__':
    app.run()


#use_reloader=True means automatic updating the server without restarting the app file
#app.run(use_reloader=True,debug=True) this is only for testing 
