# Spatio Temporal Mobile Traffic Forecasting

This is the source code of the Spatio Temporal Mobile Traffic Forecasting project done as a Master's dissertation project by Džiugas Vyšniauskas in the University of Edinburgh.

The problem tackled here can be loosely stated as:  
How can one predict the upcoming mobile internet traffic in a city, given a sequence of city-wide (geographical) traffic measurements leading to the prediction moment?

It turns out, that predicting city-wide mobile internet usage volume is similar to video prediction. Led by this idea, the following Deep Learning architectures were implemented and evaluated on Telecom Italia Big Data challenge data set (it looks as if the data set is no longer available in the original website, however it might be available from other sources):

* Sequence to Sequence LSTM
* Sequence to Sequence ConvLSTM  (convolutional LSTM)
* CNN-ConvLSTM (model combining convolutional and ConvLSTM layers)
* CNN-ConvLSTM+Attention (CNN-ConvLSTM combined with an Attention mechanism)
* PredRNN++ (an existing video prediction model: https://arxiv.org/abs/1804.06300)

For more details please refer to `thesis.pdf`.

To train a model run:
```
python experiments/experiment_runner.py --model_name <choose model> [other model parameters]
```
Note, that in this repository only a mini training set is included.  
For other available parameters see `experiments/arg_extractor.py`

To evaluate a trained model run:
```
python experiments/model_evaluator.py --model_name <model name> --model_file <path to saved model> [other model parameters]
```
