
# coding: utf-8

# In[4]:

import pandas as pd
from flask import Flask, jsonify, request
import pickle



# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

print('Outside predict')

# routes
@app.route('/', methods=['POST'])

def predict():
    print('predict_start')
    # get data
    data = request.get_json(force=True)
    print('predict_start_data')
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    print('predict_start_dataframe')
    # predictions
    result = model.predict(data_df)
    print('Before_output')
    # send back to browser
    output = {'results': int(result[0])}
    print(result)
    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True,use_reloader=False)
    
    


# In[ ]:



