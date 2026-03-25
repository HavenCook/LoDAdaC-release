import requests
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import os
import tiktoken
from datasets import load_dataset
from torch.utils.data import TensorDataset, ConcatDataset
import pickle
from itertools import islice

def read_datasets(dataset_name, data_dir=None, device='cpu'):
    if dataset_name in ["CIFAR10", "FashionMNIST", "Shakespeare", "OpenWebText"]:
        pass
    else:
        print('New dataset, readdatasets need adjustment')
        return None, None

    if data_dir==None:
        #data_dir = './data/' + dataset_name + '/'
        data_dir = os.getcwd() + '/data/' + dataset_name + '/'
        os.makedirs(data_dir, exist_ok=True)
        
    if dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
                   
        test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        return  train_dataset, test_dataset
 
    if dataset_name == "CIFAR10":
    
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        
        return train_dataset, test_dataset
     
    if dataset_name == "Shakespeare":
        # download the tiny shakespeare dataset
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            response = requests.get(data_url)
            with open(input_file_path, 'w') as f:
                f.write(response.text)

        with open(input_file_path, 'r') as f:
            data = f.read()

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }

        def encode(s):
            return [stoi[c] for c in s] # encoder: take a string, output a list of integers
        def decode(l):
            return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        # encode both to integers
        train_ids = encode(train_data)
        val_ids = encode(val_data)

        # convert to tensors

        block_size = 128 # always change it to what is being specified in the config file
        train_data = [train_ids[i:i+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        train_targets = [train_ids[i+1:i+1+block_size] for i in range(0, len(train_ids) - block_size, block_size)]
        test_data = [val_ids[i:i+block_size] for i in range(0, len(val_ids) - block_size, block_size)]
        test_targets = [val_ids[i+1:i+1+block_size] for i in range(0, len(val_ids) - block_size, block_size)]

        train_data = torch.tensor(train_data, dtype=torch.long)
        train_targets = torch.tensor(train_targets, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.long)
        test_targets = torch.tensor(test_targets, dtype=torch.long)

        # wrap in TensorDataset
        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)


        # Saving meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)

        return train_dataset, test_dataset

    # if dataset_name == "OpenWebText":
    #     # Load the dataset
    #     dataset = load_dataset("openwebtext", num_proc=4, trust_remote_code=True)

    #     # Create train/val split
    #     split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    #     train_data = split_dataset['train']
    #     val_data = split_dataset['test']

    #     # Tokenize the data
    #     enc = tiktoken.get_encoding("gpt2")
    #     def tokenize(examples):
    #         return {'ids': [enc.encode_ordinary(text) + [enc.eot_token] for text in examples['text']]}

    #     train_tokenized = train_data.map(tokenize, remove_columns=['text'], num_proc=4)
    #     val_tokenized = val_data.map(tokenize, remove_columns=['text'], num_proc=4)
    #     print("Loading openwebtext data")
    #     # Convert to tensors
    #     block_size = 1024
    #     def prepare_data(tokenized_data):
    #         all_ids = torch.tensor([item['ids'][:block_size] + [0] * (block_size - len(item['ids'])) 
    #                                 for item in tokenized_data], dtype=torch.long)
    #         return TensorDataset(all_ids[:, :-1], all_ids[:, 1:])

    #     train_dataset = prepare_data(train_tokenized)
    #     test_dataset = prepare_data(val_tokenized)

    #     return train_dataset, test_dataset
    if dataset_name == "OpenWebText":
        save_dir = os.path.join(data_dir, 'tokenized_openwebtext') #os.path.expanduser("~/AIRC/IBM_AIRC_LLM/CompOptim/data/OpenWebText/tokenized_openwebtext")
        os.makedirs(save_dir, exist_ok=True)

        val_chunk_path = os.path.join(save_dir, "val_data.pt")
        train_chunk_prefix = os.path.join(save_dir, "train_chunk_")
        chunk_size = 10_000  # Number of samples per saved chunk
        max_train_chunks = 10  # Limit total chunks to avoid overfilling home dir
        block_size = 1024

        enc = tiktoken.get_encoding("gpt2")

        def encode_text(text):
            ids = enc.encode_ordinary(text) + [enc.eot_token]
            ids = ids[:block_size] + [0] * (block_size - len(ids)) if len(ids) < block_size else ids[:block_size]
            return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)

        # Load if already tokenized
        if os.path.exists(val_chunk_path) and any(f.startswith("train_chunk_") for f in os.listdir(save_dir)):
            print("Loading tokenized OpenWebText from disk...")
            
            # Load val
            val_pairs = torch.load(val_chunk_path)
            val_dataset = TensorDataset(torch.stack([x for x, _ in val_pairs]), torch.stack([y for _, y in val_pairs]))
            
            # Load all train chunks
            train_datasets = []
            for i in range(max_train_chunks):
                chunk_path = f"{train_chunk_prefix}{i:03d}.pt"
                if os.path.exists(chunk_path):
                    pairs = torch.load(chunk_path)
                    xs, ys = zip(*pairs)
                    train_datasets.append(TensorDataset(torch.stack(xs), torch.stack(ys)))
            train_dataset = ConcatDataset(train_datasets)

        else:
            print("Streaming and tokenizing OpenWebText...")

            # Streaming dataset
            stream = load_dataset("openwebtext", streaming=True, trust_remote_code=True)["train"]

            # Create validation set
            val_size = 1000
            val_data = list(islice(stream, val_size))
            val_pairs = [encode_text(entry["text"]) for entry in val_data]
            torch.save(val_pairs, val_chunk_path)
            val_dataset = TensorDataset(torch.stack([x for x, _ in val_pairs]), torch.stack([y for _, y in val_pairs]))

            # Stream training data
            chunk = []
            chunk_id = 0
            for i, entry in enumerate(stream):
                x, y = encode_text(entry["text"])
                chunk.append((x, y))
                if (i + 1) % chunk_size == 0:
                    torch.save(chunk, f"{train_chunk_prefix}{chunk_id:03d}.pt")
                    print(f"Saved training chunk {chunk_id}")
                    chunk = []
                    chunk_id += 1
                    if chunk_id >= max_train_chunks:
                        print("Max train chunk limit reached.")
                        break
            if chunk:
                torch.save(chunk, f"{train_chunk_prefix}{chunk_id:03d}.pt")
                print(f"Saved final training chunk {chunk_id}")

            # Load them for return
            train_datasets = []
            for i in range(chunk_id + 1):
                pairs = torch.load(f"{train_chunk_prefix}{i:03d}.pt")
                xs, ys = zip(*pairs)
                train_datasets.append(TensorDataset(torch.stack(xs), torch.stack(ys)))
            train_dataset = ConcatDataset(train_datasets)

        return train_dataset, val_dataset