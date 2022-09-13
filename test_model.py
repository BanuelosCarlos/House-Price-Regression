import joblib
import numpy as np

filename = './house-price-model.pkl'
# Load the model from the file
loaded_model = joblib.load(filename)

# Create a numpy array containing a new observation 
X_new = np.array([
    [2013.167, 16.2, 289.3248, 5, 24.98203, 121.54348],
    [2013.000, 13.6, 4082, 0, 24.94155, 121.50381]]).astype('float64')

print ('New sample: {}'.format(list(X_new[0])))

# Use the model to predict 
results = loaded_model.predict(X_new)
for result in results:
    print('Prediction: {:.0f} price per unit'.format(result))



