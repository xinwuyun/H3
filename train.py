import argparse
import torch
import os
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2Config
from src.models.ssm.h3 import H3
from src.models.ssm_seq import SSMLMHeadModel
from torch.utils.tensorboard import SummaryWriter

from flash_attn.utils.generation import InferenceParams
import torch
from torch.nn import CrossEntropyLoss

from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding, TrainerCallback
from datasets import load_dataset

from loguru import logger
from functools import partial
import time

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

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

def encode_examples(example, tokenizer=None, seq_length=1024):
    # This function will be applied to each example in the dataset
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=seq_length)
    # return tokenizer(example['text'])

def load_openwebtext(tokenizer, seq_length):
    dataset = load_dataset("openwebtext/openwebtext.py", split="train")
    dataset = dataset.map(partial(encode_examples, tokenizer=tokenizer, seq_length=seq_length), batched=True, num_proc=64, remove_columns=['text'])
    return dataset

class H3Trainer(Trainer):
    def __init__(self, writer, **kwargs):
        self.writer = writer
        Trainer.__init__(self, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.labels.to(logits.device)
        # compute custom loss (suppose one has 3 labels with different weights)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_logits = logits
        # shift_labels = labels
        # Flatten the tokens
        # print(shift_logits, shift_labels)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.writer.add_scalar('Loss/Train', loss.item())
        return (loss, outputs) if return_outputs else loss

def main(args):
    torch.random.manual_seed(0)
    device = 'cuda:0'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = RandomSequenceDataset(tokenizer=tokenizer, seq_length=args.seq_length, num_samples=1000)
    # dataset = load_openwebtext(tokenizer=tokenizer, seq_length=args.seq_length)

    # build model
    d_model = args.dmodel
    n_layer = args.nlayer
    ssm_cfg = dict(mode='diag', measure='diag-lin', use_fast_fftconv=args.use_fast_fftconv)
    attn_layer_idx = args.attn_layer_idx
    attn_cfg = dict(num_heads=args.nheads)
    model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg, max_position_embeddings=args.seq_length,
                       pad_vocab_size_multiple=8).to(device=device)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # build huggingface trainer
    training_args = TrainingArguments(
        output_dir='./model_dump/',
        num_train_epochs=1,
        warmup_steps=10000,
        learning_rate=6e-4,
        per_device_train_batch_size=16,
        save_steps=500,
        save_total_limit=5,
        logging_steps=10,
        remove_unused_columns=False,
        weight_decay=0
    )
    
    writer = SummaryWriter(os.path.join('./tb', datetime.now().strftime("%m-%d-%Y-%H:%M:%S")))
    writer.add_text("config", f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, args: {args}, training_args: {training_args}")
    trainer = H3Trainer(
        writer=writer,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=dataset,
    )
    
    start = time.perf_counter()
    if args.use_profiler:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], 
                                    # schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=2, repeat=2),
                                    # schedule=torch.profiler.schedule(skip_first=1),
                                    on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
                                    profile_memory=True,
                                    with_stack=True,
                                    record_shapes=True) as prof:
            
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='H3 generation benchmarking')
    parser.add_argument('--dmodel', type=int, default=2048)
    parser.add_argument('--nlayer', type=int, default=24)
    parser.add_argument('--attn-layer-idx', type=list, default=[8, 16])
    parser.add_argument('--nheads', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seq_length', type=int, default=8192)
    parser.add_argument('--use_fast_fftconv', action='store_true')
    parser.add_argument('--use_profiler', action='store_true')
    args = parser.parse_args()
    main(args)