import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


class MultipleCompare:
    def __init__(self, file_list, file_type, file_type_list, output_name, titles):
        self.file_list = file_list
        self.file_type = file_type
        self.file_type_list = file_type_list
        self.fig, self.axs = plt.subplots(2, figsize=(8, 10))
        self.results_dir = os.path.dirname(file_list[0])
        self.compare_dir = os.path.join(os.path.dirname(self.results_dir), 'figs')
        self.output_name = output_name
        self.titles = titles

    def visualize_results(self):
        epochs = self.get_min_epochs(self.file_list, self.file_type_list)
        
        for i in range(len(self.file_list)):
            data = self.read_data(self.file_list[i], self.file_type_list[i])
            avg_test_acc, avg_test_loss, avg_train_acc, avg_train_loss = self.calculate_averages(data)
            self.axs[0].plot(range(epochs), avg_test_acc[0:epochs], label=self.file_list[i].split('-')[1])
            self.axs[1].plot(range(epochs), avg_train_loss[0:epochs], label=self.file_list[i].split('-')[1])


        self.axs[0].set_title(self.titles[0])
        self.axs[0].set_xlabel('Epochs')
        self.axs[0].set_ylabel('Accuracy')
        self.axs[0].legend()

        self.axs[1].set_title(self.titles[1])
        self.axs[1].set_xlabel('Epochs')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].legend()

        # Save the plot
        filename = '{}.png'.format(self.output_name)
        plt.savefig(filename, bbox_inches='tight')
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
                            data[rank] = {"hours": [], "test_loss": [], "test_acc": [], "train_loss": [], "train_acc": []}
                        data[rank]["hours"].append(float(l[2]))
                        data[rank]["train_loss"].append(float(l[4]))
                        data[rank]["train_acc"].append(float(l[5]))
                        data[rank]["test_loss"].append(float(l[6]))
                        data[rank]["test_acc"].append(float(l[7]))
        else:
            raise ValueError(f"Invalid file type: {file_type}")
        return data

    def get_min_epochs(self, file_list, file_type_list):
        epochs = 9999999
        for i in range(len(file_list)):
            data = self.read_data(self.file_list[i], self.file_type_list[i])
            cur_epochs = len(data[0]["test_acc"])
            if cur_epochs < epochs:
                epochs = cur_epochs
        
        return epochs
    
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
    
