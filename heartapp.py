#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('heart.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('heartindex.html')

@app.route('/predict',methods=['POST'])


def predict():
    
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    if int(output)==0:
        output="normal"
    else:
        output="abnormal"

    return render_template('heartindex.html', prediction_text='Patient must be  {}'.format(output))



        
if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




