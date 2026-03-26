from .gpt_shakespeare import GPT, GPTConfig
import os
import pickle
import glob
import torch

__all__=['nanoGPT']

exec(open('./config/train_shakespeare_char.py').read())

# attempt to derive vocab_size from the dataset
data_dir = './data/Shakespeare'
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# Check if init_from is defined in the config file
init_from = globals().get('init_from', 'scratch')
out_dir = "./results/"+globals().get('out_dir')

if init_from == 'scratch':
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=dropout
    )
    gptconf = GPTConfig(**model_args)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Find the most recent checkpoint file
    checkpoint_dirs = [os.path.join(out_dir, d) for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    checkpoint_files = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoint_dir, "chk_*.pt")))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Loading checkpoint from: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint)
        checkpoint_model_args = checkpoint['model_args']
        model_args = dict()
        # Force these config attributes to be equal, otherwise we can't resume training
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        
        # Create the model
        gptconf = GPTConfig(**model_args)

elif init_from.startswith('gpt2'):
    # initialize from OpenAI GPT-2 weights
    model_args = {}
    override_args = dict(dropout=dropout)
    temp_model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(temp_model.config, k)
    gptconf = GPTConfig(**model_args)
else:
    raise ValueError(f"Unknown init_from: {init_from}. Expected 'scratch','resume' or 'gpt2-*'")

new_out_dir = "./results/"+globals().get('out_dir')

class nanoGPT(GPT):
    def __init__(self):
        super().__init__(gptconf)
        self.config = gptconf
        self.out_dir = new_out_dir
        if block_size < self.config.block_size:
            self.crop_block_size(block_size)
            model_args['block_size'] = block_size
        self.model_args = model_args