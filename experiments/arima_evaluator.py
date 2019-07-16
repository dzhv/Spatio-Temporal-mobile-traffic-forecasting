import os
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import time

def nrmse(targets, predictions):
    targets = np.squeeze(targets)
    predictions = np.squeeze(predictions)

    error = predictions - targets
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)
    return rmse / np.mean(targets)

def predict_sequence(save_path, eval_data, order, input_start, input_size, output_size, 
    x_coord, y_coord):
    trained_model = sm.load(f"{save_path}/{x_coord}_{y_coord}.pickle")
    model = SARIMAX(eval_data[:, x_coord, y_coord], order=order)
    model_fit = model.filter(trained_model.params)
    
    prediction_wrapper = model_fit.get_prediction(start=input_start, 
                             end=input_start + input_size + output_size - 1, dynamic=input_size)
    
    return prediction_wrapper.predicted_mean[-output_size:]

def make_prediction(save_path, eval_data, order, input_start, input_size, output_size):
    grid_size = 100
    prediction = np.zeros((output_size, grid_size, grid_size))
    for x_coord in range(grid_size):
        print(f"x_coord: {x_coord}")
        for y_coord in range(grid_size):
#             print(f"pred mean: {prediction_wrapper.predicted_mean}")
            pred = predict_sequence(save_path, eval_data, order, input_start, input_size, output_size,
                x_coord, y_coord)
            prediction[:, x_coord, y_coord] += pred
#             print(prediction[:, x_coord, y_coord])
            
    return prediction

def evaluate(save_path, eval_data, order, output_size=12, input_size=12,
             train_mean=67.61768898039853, train_std=132.47248595705986):
#     input_size = order[0]  # p is input_size
    errors = []
    for i in range(0, len(eval_data) - output_size - input_size, 20):
        predictions = make_prediction(save_path, eval_data, order, i, input_size, output_size)
        targets = eval_data[i+input_size:i+input_size+output_size]
        
        predictions = predictions * train_std + train_mean 
        targets = targets * train_std + train_mean
        
        error = nrmse(targets, predictions)
        errors.append(error)
        print(f"error: {error}")
        print(f"mean error: {np.array(errors).mean()}")
                
            
    print(f"mean error: {np.array(errors).mean()}")
    print(f"error std: {np.array(errors).std()}")
    return errors

def prediction_analysis(save_path, output_path, data, order, output_size, input_size, 
    segment_start, x_coord, y_coord):
    print(f"predicting..")
    predictions = predict_sequence(save_path, data, order, segment_start, input_size, output_size,
        x_coord, y_coord)

    path = output_path + "/predictions.npy"
    print(f"saving predictions to: {path}")
    np.save(path, predictions)

print("loading data")
val = np.load("data/val.npy")
test = np.load("data/test.npy")
order = (1,0,0)
model_path = f"results/arima/p{order[0]}_d{order[1]}_q{order[2]}" 
save_path = model_path + "/saved_models"

# evaluate(save_path, val, order=(3,1,2))
prediction_analysis(save_path, model_path, test, order, output_size=30, 
    input_size=12, segment_start=50, x_coord=49, y_coord=58)

    

            