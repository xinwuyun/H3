import argparse
import torch

from transformers import GPT2Tokenizer, GPT2Config
from src.models.ssm.h3 import H3
from src.models.ssm_seq import SSMLMHeadModel

from flash_attn.utils.generation import InferenceParams
import torch
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

from loguru import logger
from functools import partial

class RandomSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_length=128, num_samples=10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random sequence of tokens
        sequence = [self.tokenizer.bos_token_id] + \
                   [torch.randint(0, len(self.tokenizer), (1,)).item() for _ in range(self.seq_length - 2)] + \
                   [self.tokenizer.eos_token_id]

        # Convert the sequence to a tensor
        input_ids = torch.tensor(sequence[:-1])
        labels = torch.tensor(sequence[1:])

        return dict(input_ids=input_ids, labels=labels)

def get_data_loader(tokenizer, batch_size=32, seq_length=128, num_samples=10000, shuffle=True):
    dataset = RandomSequenceDataset(tokenizer=tokenizer, seq_length=seq_length, num_samples=num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def encode_examples(example, seq_length=1024):
    # This function will be applied to each example in the dataset
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=seq_length)

def load_openwebtext(seq_length):
    dataset = load_dataset("openwebtext/openwebtext.py", split="train")
    dataset = dataset.map(partial(encode_examples, seq_length=seq_length), batched=True)
    return dataset

def main(args):
    torch.random.manual_seed(0)
    device = 'cuda:0'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # dataloader = get_data_loader(tokenizer, batch_size=32, seq_length=args.seq_length, num_samples=10000, shuffle=True)
    dataset = load_openwebtext()

    d_model = args.dmodel
    n_layer = args.nlayer
    ssm_cfg = dict(mode='diag', measure='diag-lin')
    attn_layer_idx = args.attn_layer_idx
    attn_cfg = dict(num_heads=args.nheads)
    model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)
    training_args = TrainingArguments(
        output_dir='./model_dump/',
        num_train_epochs=1,
        warmup_steps=200,
        learning_rate=0.001,
        per_device_train_batch_size=16,
        remove_unused_columns=False,
        weight_decay=0
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='H3 generation benchmarking')
    parser.add_argument('--dmodel', type=int, default=2048)
    parser.add_argument('--nlayer', type=int, default=24)
    parser.add_argument('--attn-layer-idx', type=list, default=[8, 16])
    parser.add_argument('--nheads', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seq_length', type=int, default=1024)
    args = parser.parse_args()
    main(args)