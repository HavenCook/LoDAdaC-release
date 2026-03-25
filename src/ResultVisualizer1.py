import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rc


#some nicematplot lib setting
SMALL_SIZE = 5
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)
fd = {'size': BIGGER_SIZE, 'family': 'serif', 'serif': ['Computer Modern']}
rc('font', **fd)
rc('text', usetex=True)


class ResultVisualizer:
    def __init__(self, filename1, filename2):
        self.data1 = self.read_data(filename1, 'yaml')
        self.data2 = self.read_data(filename2, 'tsv')
        self.results_dir = os.path.dirname(filename1)
        self.compare_dir = os.path.join(os.path.dirname(self.results_dir), 'figs')
        self.num_ranks1 = len(self.data1)
        self.num_ranks2 = len(self.data2)

    def read_data(filename, file_type):
        if file_type == 'yaml':
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
        elif file_type == 'tsv':
            data = {}
            with open(filename) as file:
                init_line = True
                for line in file:
                    l = line.split("\t")
                    if init_line == True:
                        init_line = False
                    else:
                        rank = int(l[1])
                        if rank not in data.keys():
                            data[rank] = {"hours":[],"test_loss":[],"test_acc":[],"train_loss":[],"train_acc":[]}
                        data[rank]["hours"].append(float(l[2]))
                        data[rank]["train_loss"].append(float(l[4]))
                        data[rank]["train_acc"].append(float(l[5]))
                        data[rank]["test_loss"].append(float(l[6]))
                        data[rank]["test_acc"].append(float(l[7]))
        else:
            raise ValueError(f"Invalid file type: {file_type}")
        return data

    def plot(self):
        # Get the number of epochs
        epochs1 = len(self.data1[0]["test_acc"])
        epochs2 = len(self.data2[0]["test_acc"])
        # Calculate average data across all ranks
        avg_test_acc1, avg_test_loss1, avg_train_acc1, avg_train_loss1 = self.calculate_averages(self.data1)
        avg_test_acc2, avg_test_loss2, avg_train_acc2, avg_train_loss2 = self.calculate_averages(self.data2)
        
        # Create the test metrics plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.set_title("FashionMNIST - Test Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.plot(range(epochs1), avg_test_acc1, label="DProxAdam")
        ax1.plot(range(epochs2), avg_test_acc2, label="DProxSGT")
        ax1.legend()

        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.plot(range(epochs1), avg_test_loss1, label="DProxAdam")
        ax2.plot(range(epochs2), avg_test_loss2, label="DProxSGT")
        ax2.legend()

        plot_filename = os.path.join(self.compare_dir, "test_metrics.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)

        # Create the train metrics plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.set_title("FashionMNIST - Train Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.plot(range(epochs1), avg_train_acc1, label="DProxAdam")
        ax1.plot(range(epochs2), avg_train_acc2, label="DProxSGT")
        ax1.legend()

        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.plot(range(epochs1), avg_train_loss1, label="DProxAdam")
        ax2.plot(range(epochs2), avg_train_loss2, label="DProxSGT")
        ax2.legend()

        plot_filename = os.path.join(self.compare_dir, "train_metrics.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)

    def calculate_averages(data):
        if not data:
            return (None, None, None, None)

        num_ranks = len(data)
        epochs = len(data[0]["test_acc"])

        avg_test_acc = []
        avg_test_loss = []
        avg_train_acc = []
        avg_train_loss = []

        for epoch in range(epochs):
            test_acc = [data[rank]["test_acc"][epoch] for rank in range(num_ranks)]
            test_loss = [data[rank]["test_loss"][epoch] for rank in range(num_ranks)]
            train_acc = [data[rank]["train_acc"][epoch] for rank in range(num_ranks)]
            train_loss = [data[rank]["train_loss"][epoch] for rank in range(num_ranks)]

            avg_test_acc.append(np.mean(test_acc))
            avg_test_loss.append(np.mean(test_loss))
            avg_train_acc.append(np.mean(train_acc))
            avg_train_loss.append(np.mean(train_loss))

        return avg_test_acc, avg_test_loss, avg_train_acc, avg_train_loss

    def visualize(self):
        self.plot()
        self.plot_time_variation()

    def plot_time_variation(self):
        # Get the number of epochs
        epochs1 = len(self.data1[0]["test_time"])
        epochs2 = len(self.data2[0]["hours"])

        # Calculate average time across all ranks
        avg_time1 = self.calculate_average_time(self.data1, epochs1, ["test_time", "train_time"])
        avg_time2 = self.calculate_average_time(self.data2, epochs2, ["hours"])

        # Create the time variation plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Time Variation with Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Time (seconds)")
        bar_width = 0.35
        bar1 = np.arange(len(avg_time1))
        bar2 = [x + bar_width for x in bar1]
        #ax.bar(range(epochs1), avg_time1, label="DProxAdam")
        #ax.bar(range(epochs2), avg_time2, label="DProxSGT")
        ax.bar(bar1, avg_time1, width=bar_width, label="DProxAdam")
        ax.bar(bar2, avg_time2, width=bar_width, label="DProxSGT")
        ax.legend()

        plot_filename = os.path.join(self.compare_dir, "time_variation.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig)

    def calculate_average_time(data, epochs, time_unit):
        if not data or epochs == 0:
            return None

        num_ranks = len(data)
        avg_time = []

        for epoch in range(epochs):
            if "hours" in time_unit:
                time_values = []
                for rank in range(num_ranks):
                    if epoch == 0:
                        time_values.append(data[rank]["hours"][epoch])
                    else:
                        time_values.append(data[rank]["hours"][epoch] - data[rank]["hours"][epoch-1])
                #time_values = [sum(data[rank][time][epoch] for time in time_unit) for rank in range(num_ranks)]
                time_values = [hours * 3600 for hours in time_values]  # convert hours to seconds
            else:
                time_values = [sum(data[rank][time][epoch] for time in time_unit) for rank in range(num_ranks)]
                #time_values = [sum(data[rank][time][:epoch+1]) for rank in range(num_ranks) for time in time_unit]
            avg_time.append(np.mean(time_values))

        return avg_time

class MultipleCompare:
    def __init__(self, file_list,file_type_list,output_name,name="",epochs=100,plot_list=['train_acc','test_acc','train_loss','test_loss'],figsize=(8,18)):
        self.file_list = file_list
        self.file_type_list = file_type_list
        self.fig, self.axs = plt.subplots(len(plot_list), figsize=figsize)
        self.results_dir = os.path.dirname(file_list[0])
        self.compare_dir = os.path.join(os.path.dirname(self.results_dir), output_name)
        self.output_name = output_name
        self.name = name
        self.epochs=epochs
        self.plot_list=plot_list


    def visualize_results(self):
        for i in range(len(self.file_list)):
            data = self.read_data(self.file_list[i], self.file_type_list[i])  #supports both YAML and tsv
            avg_test_acc, avg_test_loss, avg_train_acc, avg_train_loss, avg_consensus_error = self.calculate_averages(data)
            epoch = self.epochs
            idx=0
            for criteria in self.plot_list:

                if criteria.lower() == "train_acc":

                    self.axs[idx].plot(range(epoch), avg_train_acc[0:epoch], label=self.file_list[i].split('-')[1])
                    self.axs[idx].set_title(self.name+' Train Accuracy')

                elif criteria.lower() == "test_acc":

                    self.axs[idx].plot(range(epoch), avg_test_acc[0:epoch], label=self.file_list[i].split('-')[1])
                    self.axs[idx].set_title(self.name+' Test Accuracy')

                elif criteria.lower() == "train_loss":

                    self.axs[idx].plot(range(epoch), avg_train_loss[0:epoch], label=self.file_list[i].split('-')[1])
                    self.axs[idx].set_title(self.name+' Train Loss')

                elif criteria.lower() == "test_loss":

                    self.axs[idx].plot(range(epoch), avg_test_loss[0:epoch], label=self.file_list[i].split('-')[1])
                    self.axs[idx].set_title(self.name+' Test Loss')

                elif criteria.lower() == "cons_error":
                    #avg_consensus_error[0]=0
                    self.axs[idx].plot(range(epoch),avg_consensus_error[0:epoch],label=self.file_list[i].split('-')[1])
                    self.axs[idx].set_title(self.name+' Consensus Error')
                    plt.yscale('log')

                self.axs[idx].set_xlabel('Epochs')
                idx=idx+1

        for i in range(len(self.plot_list)):
            self.axs[i].legend()
        
        self.fig.tight_layout()
        plt.savefig(self.compare_dir, bbox_inches='tight')
        plt.close(self.fig)
        

    def read_data(self, filename, file_type):
        if file_type == 'yaml':
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
        elif file_type == 'tsv':
            data = {}
            with open(filename) as file:
                init_line = True
                for line in file:
                    l = line.split("\t")
                    if init_line:
                        init_line = False
                    else:
                        rank = int(l[1])
                        if rank not in data.keys():
                            data[rank] = {"hours": [], "test_loss": [], "test_acc": [], "train_loss": [], "train_acc": [], "cons_error": []}
                        data[rank]["hours"].append(float(l[2]))
                        data[rank]["train_loss"].append(float(l[4]))
                        data[rank]["train_acc"].append(float(l[5]))
                        data[rank]["test_loss"].append(float(l[6]))
                        data[rank]["test_acc"].append(float(l[7]))
                        data[rank]["cons_error"].append(float(l[3]))
        else:
            raise ValueError(f"Invalid file type: {file_type}")
        return data

    @staticmethod
    def calculate_averages(data):
        if not data:
            return (None, None, None, None)

        num_ranks = len(data)
        epochs = len(data[0]["test_acc"])

        avg_test_acc = []
        avg_test_loss = []
        avg_train_acc = []
        avg_train_loss = []
        avg_consensus_error = []

        for epoch in range(epochs):
            test_acc = [data[rank]["test_acc"][epoch] for rank in range(num_ranks)]
            test_loss = [data[rank]["test_loss"][epoch] for rank in range(num_ranks)]
            train_acc = [data[rank]["train_acc"][epoch] for rank in range(num_ranks)]
            train_loss = [data[rank]["train_loss"][epoch] for rank in range(num_ranks)]
            cons_error = [data[rank]["cons_error"][epoch] for rank in range(num_ranks)]

            avg_test_acc.append(np.mean(test_acc))
            avg_test_loss.append(np.mean(test_loss))
            avg_train_acc.append(np.mean(train_acc))
            avg_train_loss.append(np.mean(train_loss))
            avg_consensus_error.append(np.mean(cons_error))

        return avg_test_acc, avg_test_loss, avg_train_acc, avg_train_loss, avg_consensus_error

    @staticmethod
    def calculate_average_time(data, epochs, time_unit):
        if not data or epochs == 0:
            return None

        num_ranks = len(data)
        avg_time = []

        for epoch in range(epochs):
            if "hours" in time_unit:
                time_values = []
                for rank in range(num_ranks):
                    if epoch == 0:
                        time_values.append(data[rank]["hours"][epoch])
                    else:
                        time_values.append(data[rank]["hours"][epoch] - data[rank]["hours"][epoch-1])
                #time_values = [sum(data[rank][time][epoch] for time in time_unit) for rank in range(num_ranks)]
                time_values = [hours * 3600 for hours in time_values]  # convert hours to seconds
            else:
                time_values = [sum(data[rank][time][epoch] for time in time_unit) for rank in range(num_ranks)]
                #time_values = [sum(data[rank][time][:epoch+1]) for rank in range(num_ranks) for time in time_unit]
            avg_time.append(np.mean(time_values))

        return avg_time