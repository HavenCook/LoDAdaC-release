from mpi4py import MPI
import numpy as np
import torch
import yaml
from .Compressor import *

'''
CommNet is a class which deals with multiple aspects of distributed models.
The class contains methods for:
- Defining a communication network.
- Sending vectors throughout the network.
- Communication scheduling.
'''

'''
NOTES:
- Need a way to easily define the topology with a type check for string.
- setup_neighbors can ONLY do looping definitions at the moment. Cannot do standard lattice.
    To add to this. We could also add a dictionary of tuples, but this is probably too much.
'''
class CommNet:

    def __init__(self,topology="ring",comms="cpu",devices=[],nvlink=False,compressor=NoneCompressor()):
        self.topology = topology
        self.data = {}
        self.recv_data = {}
        self.devices = devices
        self.nvlink = nvlink
        self.compressor = compressor

        if comms == "cpu" or comms == "gpu":
            self.comms = comms
        else:
            print("WARNING: Invalid 'comms' argument in CommNet initialization ({})".format(comms))
        self.setup_neighbors()

    '''
    Performs the original setup of the network neighbors and
    determines the number of nodes in the network.
    '''
    def setup_neighbors(self):

        # Set up the COMM_WORLD for cpu communication (Always performed)
        self.COMM = MPI.COMM_WORLD
        self.rank = self.COMM.Get_rank()
        self.nprocs = self.COMM.Get_size()

        # Based on self.topology and self.nprocs, get the neighbors.
        self.neighbors = []
        if self.topology == "ring":
            values = [-1,1]
            for value in values:
                self.neighbors.append((self.rank + value) % self.nprocs)
                self.recv_data[(self.rank + value) % self.nprocs] = {}

        # If the number of ranks is different than the number of GPU, and we are not
        # doing CPU-only comms, then we throw a warning.
        if self.comms == "gpu":
            if self.nprocs > (len(self.devices)):
                print("WARNING: In CommNet.setup_neighbors() :: self.nprocs ({}) != GPU-count ({}).".format(self.nprocs,len(self.devices)))

    '''
    Prints all data on our current rank as well as the rank itself.
    '''
    def print_rank(self):
        print( "RANK: {} -- DATA: {} -- RECVDATA: {}".format(self.rank,self.data,self.recv_data) )

    '''
    Basic send and receive calls.
    '''
    def net_send(self,neighbor_id,field,name,tag=0,verbose=False):

        if self.comms == "cpu":

                compressed_tensor_info = self.compressor.compress(self.data[field][name])
                self.COMM.send(compressed_tensor_info[1:],neighbor_id,tag=tag) # send the rest of the compressed_tensor


        elif self.comms == "gpu":

            if self.nvlink:

                # Utilize cuda aware MPI.
                compressed_tensor_info = self.compressor.compress(self.data[field][name])
                self.COMM.send(compressed_tensor_info[1:],neighbor_id,tag=tag)

            else:

                # Get a send tensor onto our compute rank.
                send_tensor = self.compressor.compress(self.data[field][name].clone().to("cpu"))

                # Send the variable over COMM_WORLD to our neighbor.
                self.COMM.send(send_tensor[1:],neighbor_id,tag=tag)

        else:

            # Print a warning.
            print("WARNING: In CommNet.net_send(), invalid 'comms' field.")

        if verbose:
            print("RANK: {} SENT-TO: {}".format(self.rank,neighbor_id) )


    def net_recv(self,neighbor_id,name,tag=0,unique=False,verbose=False):

        if self.comms == "cpu":

            res = None

            if unique:

                tensor_information=self.COMM.recv(source=neighbor_id,tag=tag)
                self.recv_data[neighbor_id][name]=self.compressor.decompress(tensor_information)

            else:


                tensor_information=self.COMM.recv(source=neighbor_id,tag=tag)
                self.recv_data["rec_field"][name] = self.compressor.decompress(tensor_information)
                self.recv_data["reduced"][name] += self.recv_data["rec_field"][name]

        elif self.comms == "gpu":

            if self.nvlink:

                # Utilize cuda aware MPI for our comms.
                tensor_information=self.COMM.recv(source=neighbor_id,tag=tag)
                self.recv_data[neighbor_id][name]=self.compressor.decompress(tensor_information)

            else:

                # Our process receives the data from our neighbor core.
                tensor_information=self.COMM.recv(source=neighbor_id,tag=tag)

                # Get our GPU ID.
                gpu_id = self.devices[self.rank % len(self.devices)]

                # Now we copy it to our GPU.
                self.recv_data["rec_field"][name] = self.compressor.decompress(tensor_information).to(gpu_id)
                self.recv_data["reduced"][name] += self.recv_data["rec_field"][name]

        else:

            # Print a warning.
            print("WARNING: In CommNet.net_recv(), invalid 'comms' field.")

        if verbose:
            print("RANK: {} RECEIVED-FROM: {}".format(self.rank,neighbor_id) )

    '''
    Considering the topology, performs a "local all-gather" that
    only works over neighbors. (I know this is nonsense wording, but you get it.)
    NOTES:
        - Currently only works with a ring network and an even self.nprocs value.
        - Assumes the data being sent and received are pytorch
            tensors of the same size.
    '''
    def neighbor_gather(self,field,name,unique=False):

        # Now we consider the topology of the network to decide
        # our communication scheme.
        if self.topology == "ring":

            # If our modulus is even.
            if self.nprocs % 2 == 0:

                # If our node is even, receive first; odd, send first.
                if self.rank % 2 == 0:
                    self.net_send(neighbor_id=(self.rank+1)%self.nprocs,field=field,name=name)
                    self.net_send(neighbor_id=(self.rank-1)%self.nprocs,field=field,name=name)
                    self.net_recv(neighbor_id=(self.rank+1)%self.nprocs,name=name,unique=unique)
                    self.net_recv(neighbor_id=(self.rank-1)%self.nprocs,name=name,unique=unique)
                else:
                    self.net_recv(neighbor_id=(self.rank-1)%self.nprocs,name=name,unique=unique)
                    self.net_recv(neighbor_id=(self.rank+1)%self.nprocs,name=name,unique=unique)
                    self.net_send(neighbor_id=(self.rank-1)%self.nprocs,field=field,name=name)
                    self.net_send(neighbor_id=(self.rank+1)%self.nprocs,field=field,name=name)

            # If our modulus is odd
            else:
                # Perform communication between the first and last rank.
                if self.rank == (self.nprocs-1):
                    self.net_recv(neighbor_id=self.nprocs-2,name=name,unique=unique)
                    self.net_send(neighbor_id=self.nprocs-2,field=field,name=name)
                    self.net_send(neighbor_id=0,field=field,name=name)
                    self.net_recv(neighbor_id=0,name=name,unique=unique)
                elif self.rank == 0:
                    self.net_send(neighbor_id=1,field=field,name=name)
                    self.net_recv(neighbor_id=1,name=name,unique=unique)
                    self.net_recv(neighbor_id=self.nprocs-1,name=name,unique=unique)
                    self.net_send(neighbor_id=self.nprocs-1,field=field,name=name)
                elif self.rank % 2 == 1:
                    self.net_recv(neighbor_id=(self.rank-1)%self.nprocs,name=name,unique=unique)
                    self.net_send(neighbor_id=(self.rank-1)%self.nprocs,field=field,name=name)
                    self.net_send(neighbor_id=(self.rank+1)%self.nprocs,field=field,name=name)
                    self.net_recv(neighbor_id=(self.rank+1)%self.nprocs,name=name,unique=unique)
                elif self.rank % 2 == 0:
                    self.net_send(neighbor_id=(self.rank+1)%self.nprocs,field=field,name=name)
                    self.net_recv(neighbor_id=(self.rank+1)%self.nprocs,name=name,unique=unique)
                    self.net_recv(neighbor_id=(self.rank-1)%self.nprocs,name=name,unique=unique)
                    self.net_send(neighbor_id=(self.rank-1)%self.nprocs,field=field,name=name)

        self.COMM.Barrier()






    '''
    Performs a neighbor gather and then reduces the results to a reduced state.
    NOTES:
        - Currently also only works with a basic choice of mixing matrix where every node
          is worth the same amount. (TO DO)
    '''


    def neighbor_reduce(self,field,name,unique=False):


        # Now take those results and average them.
        # If we want to store each result as it comes in...
        if unique:

            # First we call neighbor_gather.
            self.neighbor_gather(field,name,unique=True)

            # Then define "reduced" so we have somewhere to store everything.
            self.recv_data["reduced"][name] = torch.zeros_like(self.data[field][name])
            self.recv_data["reduced"][name] += self.data[field][name]
            for key in self.recv_data.keys():
                if key != "reduced" and key != "rec_field":
                    self.recv_data["reduced"][name] += self.recv_data[key][name]
            self.recv_data["reduced"][name] = self.recv_data["reduced"][name].div((len(self.neighbors)+1))

        # If we want to only store one result at a time, and reduce it as we go.
        else:
            self.recv_data["reduced"][name] = torch.zeros_like(self.data[field][name])
            self.recv_data["rec_field"][name] = torch.zeros_like(self.data[field][name])
            self.neighbor_gather(field,name,unique=False)
            self.recv_data["reduced"][name] += self.data[field][name]
            self.recv_data["reduced"][name] = self.recv_data["reduced"][name].div((len(self.neighbors)+1))

        # Just put a barrier here to be safe.
        self.COMM.Barrier()
        return self.recv_data["reduced"][name],self.data[field][name]

    '''
    Does an all reduce over our network.
    '''
    def all_reduce(self,field,name):

        # Perform an allgather.
        collected_data = self.COMM.allgather(self.data[field][name])
        self.recv_data["reduced"][name] = torch.zeros_like(self.data[field][name])
        for node in range(self.nprocs):
            self.recv_data["reduced"][name] += collected_data[node]/self.nprocs

        self.COMM.Barrier()

        return self.recv_data["reduced"][name],self.data[field][name]