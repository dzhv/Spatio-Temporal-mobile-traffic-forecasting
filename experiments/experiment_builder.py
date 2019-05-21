import tqdm
import os
import numpy as np
import time
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from storage_utils import save_statistics
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

        # self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
        #                             weight_decay=weight_decay_coefficient)

        # Generate the directory names

        self.experiment_folder = os.path.join(parent_folder, "results", experiment_name)
        self.experiment_logs = os.path.join(self.experiment_folder, "result_outputs")
        self.experiment_saved_models = os.path.join(self.experiment_folder, "saved_models")
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = float("inf")

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        # save args
        with open(os.path.join(self.experiment_folder, "arguments.txt"), "w") as file:
            file.write(str(args))

        self.train_mean = args.train_mean
        self.train_std = args.train_std

        self.num_epochs = num_epochs
        self.continue_from_epoch = continue_from_epoch
        # self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()        

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss.
        :param x: The inputs to the model.
        :param y: The targets for the model.
        :return: the loss for this batch
        """
        self.model.train_mode()

        loss = self.model.train(x, y)

        # self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        # loss.backward()  # backpropagate to compute gradients for current iter loss

        # self.optimizer.step()  # update network parameters

        # return loss.data.detach().cpu().numpy()
        
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
        predictions = out * self.train_std + self.train_mean 
        targets = y * self.train_std + self.train_mean

        if len(y.shape) == 1 or y.shape[-1] == 1:     # if this is a 1 step prediction
            nrmse_loss = nrmse(targets, predictions)
        else:                                   # if this is a multi step prediction
            nrmse_loss = nrmse(targets[:, -1], predictions[:, -1])

        print(f"mse loss: {mse_loss}")
        print(f"nrmse loss: {nrmse_loss}")

        return mse_loss, nrmse_loss
        

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        path = os.path.join(model_save_dir, f"{model_save_name}_{model_idx}")
        self.model.save(path)

        # state['network'] = self.state_dict()  # save network parameter and other variables.
        # torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
        #     model_idx))))  # save state at prespecified filepath

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
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_loss'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model 
            and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        if self.continue_from_epoch == -1:
            self.model.reset_parameters()


        # initialize a dict to keep the per-epoch metrics        
        total_losses = {"train_loss": [], "val_loss": [], "val_nrmse_loss": [], "curr_epoch": []}  
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
            # if current epoch's mean val acc is greater than the saved best val acc then
            if val_mean_loss < self.best_val_model_loss:  
                # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_loss = val_mean_loss  
                # set the experiment-wise best val idx to be the current epoch's idx
                self.best_val_model_idx = epoch_idx  

            # get mean of all metrics of current epoch metrics dict, 
            # to get them ready for storage and output on the terminal.
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))  

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)

            # How to load a csv file if you need to
            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') 

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_loss'] = self.best_val_model_loss
            self.state['best_val_model_idx'] = self.best_val_model_idx

            # save model and best val idx and best val acc, using the model dir, model name and model idx
            self.save_model(model_save_dir=self.experiment_saved_models,                
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            # save model and best val idx and best val acc, using the model dir, model name and model idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx='latest', state=self.state)

        # print("Generating test set evaluation metrics")
        # self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
        #                 # load best validation model
        #                 model_save_name="train_model")
        # current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict
        # with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
        #     for x, y in self.test_data:  # sample batch
        #         # compute loss and accuracy by running an evaluation step
        #         loss, accuracy = self.run_evaluation_iter(x=x, y=y)  
        #         current_epoch_losses["test_loss"].append(loss)  # save test loss
        #         current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
        #         pbar_test.update(1)  # update progress bar status
        #         pbar_test.set_description(
        #             "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output

        # test_losses = {key: [np.mean(value)] for key, value in
        #                current_epoch_losses.items()}  # save test set metrics in dict format
        # save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
        #                 # save test set metrics on disk in .csv format
        #                 stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        # return total_losses, test_losses
