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
    # start = time.time()
    trained_model = sm.load(f"{save_path}/{x_coord}_{y_coord}.pickle")
    # post_load = time.time()
    model = SARIMAX(eval_data[input_start:input_start+input_size, x_coord, y_coord], order=order)
    # post_create = time.time()
    model_fit = model.filter(trained_model.params)
    # post_filter = time.time()
    
    prediction_wrapper = model_fit.get_prediction(start=0, 
                             end= input_size + output_size - 1, dynamic=input_size)

    post_predict = time.time()

    # print(f"loading time: {post_load - start}")
    # print(f"create time: {post_create - post_load}")
    # print(f"filter time: {post_filter - post_create}")
    # print(f"predict time: {post_predict - post_filter}")
    # print(f"full time: {post_predict - start}") 

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

def calculate_error(predictions, targets, train_mean, train_std):
    predictions = predictions * train_std + train_mean 
    targets = targets * train_std + train_mean

    return nrmse(targets, predictions)

def evaluate(save_path, eval_data, order, output_size=12, input_size=12,
             train_mean=67.61768898039853, train_std=132.47248595705986):
#     input_size = order[0]  # p is input_size
    errors_10 = []
    errors_12 = []
    errors_30 = []
    indexes = range(0, len(eval_data) - output_size - input_size, 20)
    for count, i in enumerate(indexes):
        predictions = make_prediction(save_path, eval_data, order, i, input_size, output_size)
        targets = eval_data[i+input_size:i+input_size+output_size]

        print(f"predictions shape: {predictions.shape}")
        
        errors_10.append(calculate_error(predictions[:10], targets[:10], train_mean, train_std))
        errors_12.append(calculate_error(predictions[:12], targets[:12], train_mean, train_std))
        errors_30.append(calculate_error(predictions[:30], targets[:30], train_mean, train_std))
        
        print(f"mean 10 step error: {np.array(errors_10).mean()}")
        print(f"mean 12 step error: {np.array(errors_12).mean()}")
        print(f"mean 30 step error: {np.array(errors_30).mean()}")
        print(f"{count}/{len(indexes)}")
            
    print(f"mean 10 step error: {np.array(errors_10).mean()}")
    print(f"error std: {np.array(errors_10).std()}")
    print(f"mean 12 step error: {np.array(errors_12).mean()}")
    print(f"error std: {np.array(errors_12).std()}")
    print(f"mean 30 step error: {np.array(errors_30).mean()}")
    print(f"error std: {np.array(errors_30).std()}")
    return (errors_10, errors_12, errors_30)

def prediction_analysis(save_path, output_path, data, order, locations, output_size, input_size):
    print(f"predicting..")

    result = {}
    for location in locations:
        cell = location['cell']  
        predictions = predict_sequence(save_path, data, order, location['from'], 
            input_size, output_size,
            cell[0], cell[1])

        result.update({ f"{location['from']}_{cell[0]}_{cell[1]}": predictions})

    path = output_path + "/predictions.npy"
    print(f"saving predictions to: {path}")
    np.save(path, result)

def fullgrid_prediction_analysis(save_path, output_path, data, order, indexes, output_size, input_size):
    print(f"predicting..")

    result = {}

    for i, index in enumerate(indexes):
        prediction = np.zeros((100, 100))
        print(f"index {i}/{len(indexes)}")
        for x in range(100):
            print(f"x: {x}")
            for y in range(100):
                predictions = predict_sequence(save_path, data, order, index, 
                    input_size, output_size, x, y)
                prediction[x, y] = predictions[-1]


            targets = data[index+input_size:index+input_size+output_size]

            result.update({ str(index): prediction})
            result.update({ f"{index}_y": targets})

    path = output_path + "/predictions_fullgrid.npy"
    print(f"saving predictions to: {path}")
    np.save(path, result)

print("loading data")
val = np.load("data/val.npy")
test = np.load("data/test.npy")
order = (12,1,2)
model_path = f"results/arima/p{order[0]}_d{order[1]}_q{order[2]}" 
save_path = model_path + "/saved_models"

# noise = np.random.randn(4000, 100, 100) * 10
# evaluate(save_path, test, order=order, output_size=30)

locations = [
    {'from': 49, 'cell': (38, 63)},
    {'from': 100, 'cell': (49, 58)},
    {'from': 650, 'cell': (47, 58)}
]
# prediction_analysis(save_path, model_path, test, order, locations, output_size=30, input_size=12)

indexes = [10, 11, 88, 221]
fullgrid_prediction_analysis(save_path, model_path, test, order, indexes, output_size=30, input_size=12)