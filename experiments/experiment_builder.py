import tqdm
import os
import numpy as np
import time
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from storage_utils import save_statistics, load_statistics, save_best_val_scores
from models.losses import mse
from models.losses import nrmse_numpy as nrmse


class ExperimentBuilder(object):
    def __init__(self, args, model, experiment_name, num_epochs, train_data, val_data,
                 continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation
         of a deep net on a given dataset. It also takes care of saving per epoch models and automatically 
         inferring the best val model to be used for evaluating the test set metrics.
        :param model: A Model class instance
        :param experiment_name: The name of the experiment.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) 
                or whether we'll reload a previously saved model of epoch 'continue_from_epoch'
                and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = model
          
        self.train_data = train_data
        self.val_data = val_data

        # Generate the experiment directories
        self.experiment_folder = os.path.join(parent_folder, "results", experiment_name)
        self.experiment_logs = os.path.join(self.experiment_folder, "result_outputs")
        self.summary_file = os.path.join(self.experiment_logs, "summary.csv")
        self.experiment_saved_models = os.path.join(self.experiment_folder, "saved_models")
        print(self.experiment_folder, self.experiment_logs)

        if not os.path.exists(self.experiment_folder):
            os.mkdir(self.experiment_folder)

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_loss = float("inf")
        self.best_val_nrmse_model_idx = 0
        self.best_val_nrmse_loss = float("inf")

        # save experiment arguments for traceability
        with open(os.path.join(self.experiment_folder, "arguments.txt"), "w") as file:
            file.write(str(args) + "\n")

        self.train_mean = args.train_mean
        self.train_std = args.train_std

        self.num_epochs = num_epochs
        self.continue_from_epoch = continue_from_epoch

        self.post_train_time = None

        if continue_from_epoch == -2:  # load the last saved model from the experiment_saved_models directory
            self.load_model(model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.load_model(model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)            
        else:
            self.starting_epoch = 0
            self.metrics = self.empty_metrics()
            

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss.
        :param x: The inputs to the model.
        :param y: The targets for the model.
        :return: the loss for this batch
        """
        self.model.train_mode()

        train_start_time = time.time()

        if not self.post_train_time is None:
            non_train_time = train_start_time - self.post_train_time
            non_train_time = "{:.4f}".format(non_train_time)
            # print(f"operations between model.train took: {non_train_time} seconds")

        loss = self.model.train(x, y)

        self.post_train_time = time.time()
        train_elapsed_time = self.post_train_time - train_start_time
        train_elapsed_time = "{:.4f}".format(train_elapsed_time)
        # print(f"model.train took: {train_elapsed_time} seconds")

        return loss

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iteration. Returns the loss.
        :param x: The inputs to the model.
        :param y: The targets for the model.
        :return: the loss for this batch
        """
        self.model.eval_mode()  # sets the system to validation mode

        out = self.model.forward(x)
        mse_loss = mse(y, out)
        print(f"mse loss: {mse_loss}")

        predictions = np.array(out) * self.train_std + self.train_mean 
        targets = y * self.train_std + self.train_mean

        nrmse_loss = nrmse(targets, predictions)        
        print(f"nrmse loss: {nrmse_loss}")

        return mse_loss, nrmse_loss

    def save_model(self, model_save_dir, model_save_name, epoch_idx):
        """
        Save the model state.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.
        """
        epoch_model_path = os.path.join(model_save_dir, f"{model_save_name}_{epoch_idx}")
        latest_model_path = os.path.join(model_save_dir, f"{model_save_name}_latest")
        self.model.save(epoch_model_path)
        self.model.save(latest_model_path)

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared 
        with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state 
                 into the system state without returning it
        """
        print(f"\nLoading model: {model_save_name} {model_idx}, from: {model_save_dir}\n")

        path = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx)))
        self.model.load(path)

        stats = load_statistics(self.summary_file)

        val_losses = np.array(stats['val_loss']).astype(np.float)
        self.best_val_loss = np.min(val_losses)
        self.best_val_model_idx = np.argmin(val_losses)
        val_nrmse_losses = np.array(stats['val_nrmse_loss']).astype(np.float)
        self.best_val_nrmse_loss = np.min(val_nrmse_losses)
        self.best_val_nrmse_model_idx = np.argmin(val_nrmse_losses)
        
        curr_epoch = int(stats['curr_epoch'][-1]) + 1
        self.starting_epoch = curr_epoch

        print(f"current epoch: {curr_epoch}")
        self.metrics = stats

    def empty_metrics(self):
        return {"train_loss": [], "val_loss": [], "val_nrmse_loss": [], "curr_epoch": []}

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model 
            and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        if self.continue_from_epoch == -1:
            self.model.reset_parameters()
        
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "val_loss": [], "val_nrmse_loss": []}

            with tqdm.tqdm(total=self.train_data.num_batches) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    loss = self.run_train_iter(x=x, y=y)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}".format(loss))

            with tqdm.tqdm(total=self.val_data.num_batches) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    mse_loss, nrmse_loss = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    current_epoch_losses["val_loss"].append(mse_loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_nrmse_loss"].append(nrmse_loss)
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}".format(loss))
            
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            # if lowest validation loss was achieved in this epoch
            if val_mean_loss < self.best_val_loss:
                self.best_val_loss = val_mean_loss
                self.best_val_model_idx = epoch_idx

            val_nrmse_mean_loss = np.mean(current_epoch_losses['val_nrmse_loss'])
            # if lowest nrmse validation loss was achieved in this epoch
            if val_nrmse_mean_loss < self.best_val_nrmse_loss:
                self.best_val_nrmse_loss = val_nrmse_mean_loss
                self.best_val_nrmse_model_idx = epoch_idx

            # get mean of all metrics of current epoch metrics dict, 
            # to get them ready for storage and output on the terminal.
            for key, value in current_epoch_losses.items():
                self.metrics[key].append(np.mean(value))

            self.metrics['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=self.metrics, current_epoch=epoch_idx,
                            continue_from_mode=(self.starting_epoch != 0 or i > 0))

            save_best_val_scores(experiment_log_dir=self.experiment_logs, filename='best_val_scores.csv',
                best_val_loss=self.best_val_loss, best_val_model_idx=self.best_val_model_idx, 
                best_val_nrmse_loss=self.best_val_nrmse_loss, best_val_nrmse_model_idx=self.best_val_nrmse_model_idx)

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

            # save the current model
            self.save_model(model_save_dir=self.experiment_saved_models,     
                            model_save_name="train_model", epoch_idx=epoch_idx)

