import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import string

class MultipleCompare:
    def __init__(self, file_list, file_type_list, output_names):
        self.file_list = file_list
        self.file_type_list = file_type_list
        self.results_dir = os.path.dirname(file_list[0])
        self.output_names = output_names
        self.marker_list = ['o','s','p','d','x']
        self.mark_incr = 8
        self.fig_size_x = 5
        self.fig_size_y = 3.5

    def visualize_results(self):
        epochs = self.get_min_epochs(self.file_list, self.file_type_list)
        # print(epochs)
        # print(self.file_list)
        # optimizers = [name.split(' - ')[1] for name in self.file_list]
        avg_loss = {}
        avg_acc = {}
        avg_cons = {}
        epochs_axis = {}
        cur_epochs = {}
        optimizers = []
        for i in range(len(self.file_list)):
            optimizer = self.file_list[i].split('/')[-1].split('_')[0]
            # optimizer = self.file_list[i].split('_')[0]
            optimizers.append(optimizer)
            print(optimizer)
            optimizer = optimizers[i]
            data = self.read_data(self.file_list[i], self.file_type_list[i])
            avg_loss[optimizer], avg_acc[optimizer], avg_cons[optimizer] = self.calculate_averages(data)
            # avg_loss[optimizer], avg_acc[optimizer] = self.calculate_averages(data)
            
            epochs_axis[optimizer] = [i for i in range(epochs)]
            if optimizer != "DAMSCo":
            #if optimizer == "NA":
                cur_epochs[optimizer] = round(epochs / 2)
                for j in range(len(epochs_axis[optimizer])):
                    epochs_axis[optimizer][j] *= 2
            else:
                cur_epochs[optimizer] = epochs
        
        x_axis_label = 'Communication Rounds'
        self.fig_loss, self.axs_loss = plt.subplots()
        init = 4
        i = 0
        for o in optimizers:   
            init = init + 2
            # print(o)
            epochs = cur_epochs[o]
            # print(epochs_axis[o][0:epochs], avg_loss[o][0:epochs])
            self.axs_loss.plot(epochs_axis[o][0:epochs], avg_loss[o][0:epochs], marker=self.marker_list[i], markevery=init, label=o)
            i += 1

        self.axs_loss.set_xlabel(x_axis_label)
        self.axs_loss.set_ylabel('Train Loss')
        self.axs_loss.legend()
        
        filename = '{}.pdf'.format(self.output_names[0])
        plt.rcParams["figure.figsize"] = (self.fig_size_x, self.fig_size_y)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(self.fig_loss)
        # print(filename)
        
        
        self.fig_acc, self.axs_acc = plt.subplots()
        init = 4
        i = 0
        for o in optimizers:   
            init = init + 2
            # print(o)
            epochs = cur_epochs[o]
            # print(epochs_axis[o][0:epochs], avg_acc[o][0:epochs])
            self.axs_acc.plot(epochs_axis[o][0:epochs], avg_acc[o][0:epochs], marker=self.marker_list[i], markevery=init, label=o)
            i += 1
        
        self.axs_acc.set_xlabel(x_axis_label)
        self.axs_acc.set_ylabel('Test Accuracy')
        self.axs_acc.legend()
        
        filename = '{}.pdf'.format(self.output_names[1])
        plt.rcParams["figure.figsize"] = (self.fig_size_x, self.fig_size_y)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(self.fig_acc)
        # print(filename)
        
        
        self.fig_cons, self.axs_cons = plt.subplots()
        init = 4
        i = 0
        for o in optimizers:   
            init = init + 1
            # print(o)
            epochs = cur_epochs[o]
            # print(epochs_axis[o][0:epochs], avg_cons[o][0:epochs])
            self.axs_cons.plot(epochs_axis[o][0:epochs], avg_cons[o][0:epochs], marker=self.marker_list[i], markevery=init, label=o)
            i += 1

        self.axs_cons.set_yscale('log')
        self.axs_cons.set_xlabel(x_axis_label)
        self.axs_cons.set_ylabel('Consensus Errorls')
        self.axs_cons.legend()
        
        filename = '{}.pdf'.format(self.output_names[2])
        plt.rcParams["figure.figsize"] = (self.fig_size_x, self.fig_size_y)
        plt.savefig(filename, bbox_inches='tight')
        plt.close(self.fig_cons)
        # print(filename)
        


    def read_data(self, filename, file_type):
        print(filename)
        if file_type == 'yaml':
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
        elif file_type == 'tsv':
            data = {}
            with open(filename) as file:
                print(filename)
                init_line = True
                for line in file:
                    l = line.split("\t")
                    if init_line:
                        init_line = False
                    else:
                        rank = int(l[1])
                        if rank not in data.keys():
                            data[rank] = {"test_acc": [], "train_loss": [], "cons_error": []}
                        data[rank]["train_loss"].append(float(l[4]))
                        data[rank]["test_acc"].append(float(l[7]))
                        data[rank]["cons_error"].append(float(l[3]))
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
        epochs = len(data[0]["train_loss"])

        avg_loss = []
        avg_acc = []
        avg_cons = []

        for epoch in range(epochs):
            train_loss = [data[rank]["train_loss"][epoch] for rank in range(num_ranks)]
            test_acc = [data[rank]["test_acc"][epoch] for rank in range(num_ranks)]
            cons_error = [data[rank]["cons_error"][epoch] for rank in range(num_ranks)]

            avg_loss.append(np.mean(train_loss))
            avg_acc.append(np.mean(test_acc))
            avg_cons.append(np.mean(cons_error))

        return avg_loss, avg_acc, avg_cons