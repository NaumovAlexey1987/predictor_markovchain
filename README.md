# predictor_markovchain

there is a predictior using markov chain to predict

to initialize the predictor use predictor_markov_chain() from main_predictor.py 

next need to fill a train set to train with fit_train_array(train_array) 

next need to create a graph with produce_graph(predict_trajectory_len, round = 1)

then you can to predict with predict(preceding_array, horizon, non_pred_model, up_method)

See main.ipynb and test.ipynb for examples
