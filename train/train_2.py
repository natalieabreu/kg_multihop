import os
import itertools
import yaml

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import numpy as np
import json
from datasets import load_dataset, IterableDataset
from torch.distributed import barrier

from custom_trainer import CustomTrainer

def load_mapping(file_path):
    """Loads the config-to-filepath mapping from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
DATA_FILE_MAP = load_mapping("/n/home07/nabreu/kg_multihop/train/data_files.json")

def get_filepath(config_or_label):
    """
    Retrieves the filepath based on either a config label (string)
    or a config dictionary of attributes (dict).
    """
    # If input is a string, assume it's a label like "config_2"
    if isinstance(config_or_label, str):
        if config_or_label in DATA_FILE_MAP:
            return get_samples_file(config_or_label)
        else:
            raise ValueError(f"No matching label '{config_or_label}' found in mappings.")

    # Otherwise, assume it's a dictionary of attributes
    elif isinstance(config_or_label, dict):
        for lbl, entry in DATA_FILE_MAP.items():
            if all(config_or_label[key] == entry[key] for key in config_or_label):
                return get_samples_file(lbl)
        raise ValueError("No matching filepath found for the given config.")
    
    else:
        raise TypeError("Input should be either a config label (str) or a config dictionary (dict).")

def get_samples_file(config_label):
    # Check if config_label exists in data_file_map
    print(f"Getting samples file for config label: {config_label}")
    if config_label not in DATA_FILE_MAP:
        raise ValueError(f"Config label {config_label} not found in data file map.")

    config = DATA_FILE_MAP[config_label]

    # Check if 'samples-only' key exists
    if 'samples-only' in config:
        # Return the 'samples-only' filepath
        samples_file = config['samples-only']
        print(f"Using existing samples-only file: {samples_file}")
    else:
        # 'samples-only' doesn't exist, create the samples file
        original_file = config['filepath']
        samples_file = original_file.replace(".json", "-samples.json")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            # Load the original data file and extract samples
            if not os.path.exists(samples_file):
                print(f"'samples-only' file not found, creating {samples_file} from {original_file}")
                with open(original_file, 'r') as f:
                    data = json.load(f)

                samples = data['samples']

                # Write the samples to the new file
                with open(samples_file, 'w') as f:
                    json.dump(samples, f)
            else:
                print(f"'samples-only' file already exists: {samples_file}")
        
        barrier()

    return samples_file

# Load config file
with open("/n/home07/nabreu/kg_multihop/train/train_2_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Get SLURM array task ID (job index)
slurm_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))

# Create a list of all possible hyperparameter combinations
hyperparams = config['hyperparams']



# train_path = config.get("train_path", "/n/home07/nabreu/kg_multihop/202409-1123-4731-samples.json") # 1M p=1
train_path = get_filepath(config.get("train_data_config"))
eval_path = config.get("eval_path", "/n/home07/nabreu/kg_multihop/eval-samples.json")
eval_query_dataset_path = config.get("eval_query_dataset_path", "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json")
streaming = config.get("streaming", False)
notes = config.get("notes", "")

data_files = {
    'train': train_path,
    'validation': eval_path
}

hyperparams = config.get("hyperparams", {})
per_device_train_batch_size = hyperparams.get("per_device_train_batch_size", 8)
per_device_eval_batch_size = hyperparams.get("per_device_eval_batch_size", 2)
eval_accumulation_steps = hyperparams.get("eval_accumulation_steps", 100)

# learning_rate = hyperparams.get("learning_rate", 0.001)
# weight_decay = hyperparams.get("weight_decay", 0.1)
# gradient_accumulation_steps = hyperparams.get("gradient_accumulation_steps", 3)
# warmup_steps = hyperparams.get("warmup_steps", 1000)
# num_train_epochs = hyperparams.get("num_epochs", 100)

param_combinations = list(itertools.product(
    hyperparams['num_epochs'],
    hyperparams['learning_rate'],
    hyperparams['weight_decay'],
    hyperparams['gradient_accumulation_steps'],
    hyperparams['warmup_steps'],
))

# Select the combination for this job based on SLURM_ARRAY_TASK_ID
chosen_combination = param_combinations[slurm_index]

# Unpack the chosen combination into individual hyperparameters
num_epochs, learning_rate, weight_decay, gradient_accumulation_steps, warmup_steps = chosen_combination

config_dict = {
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "warmup_steps": warmup_steps,
    "train_path": train_path,
    "eval_path": eval_path,
    "eval_query_dataset_path": eval_query_dataset_path,
    "notes": notes
}

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    print(f"SLURM task ID: {slurm_index}")
    print(config_dict)

# Load GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenizer settings
tokenizer.pad_token = tokenizer.eos_token

# Load custom dataset from a JSON file
# Assuming your JSON file is at './data/train.json' and has a list of text samples
# data_files = {
#     # "train": "/n/home07/nabreu/kg_multihop/202409-1123-4731-samples.json",     # 1M p=1
#     # "train": "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/text_only/202409-1608-3521.json", # 10M p=1
#     # "train": "/n/home07/nabreu/kg_multihop/train-100-samples.json",
#     # "train": "/n/home07/nabreu/kg_multihop/unique-samples-1M.json", #~800k samples p=1 
#     # "train": "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/denoising_1000000_outneighbors_all/202410-1521-1129-samples.json", # random+all relation templates
#     # "train": "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/denoising_1000000_outneighbors_spo/202410-1521-1013-samples.json", # random+spo relation templates
#     # "validation": "/n/home07/nabreu/kg_multihop/unique-samples-eval-100.json"
#     # "validation": "/n/home07/nabreu/kg_multihop/eval-100-samples.json",
#     "train": "/n/home07/nabreu/kg_multihop/eval-samples.json",  # Path to your validation data (optional)
#     "validation": "/n/home07/nabreu/kg_multihop/eval-samples.json"  # Path to your validation data (optional)

# }

# eval_query_dataset_path = "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json" # fictional entities


# data_files = { # original entity files
#     'train': '/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/denoising_1000000_outneighbors_orig_entities/202410-1614-4450-samples.json',
#     'validation': '/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_orig_entities-samples.json',
# }
# eval_query_dataset_path="/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries_orig_entities.json" # original entities


# streaming=False
'''
Non-concatenated dataset
'''
# dataset = load_dataset('json', data_files=data_files)
# # dataset.with_format("torch")

# # Preprocess the dataset
# def tokenize_function(examples):
#     # Tokenize the input texts and shift them to prepare for next-token prediction
#     # Padding and truncating all examples to a max length
#     tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
#     # For causal language modeling, the labels are simply the input_ids shifted by one position
#     tokenized["labels"] = tokenized["input_ids"].copy()
    
#     return tokenized

# # Tokenize the datasets
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

'''
Concatenated dataset
'''
def concatenate_and_tokenize_all(dataset):
    # List to store tokenized sequences
    all_input_ids = []
    
    # Iterate over all examples in the dataset
    count = 0
    for example in dataset:
        # Tokenize each individual text
        tokenized = tokenizer(example["text"], truncation=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze().tolist()
        all_input_ids.extend(input_ids)
        
        # Insert EOS token after each text
        all_input_ids.append(tokenizer.eos_token_id)
        count += 1
    print(count)

    # Now chunk it into sequences of length 512
    chunked_input_ids = [
        all_input_ids[i:i + 512]
        for i in range(0, len(all_input_ids), 512)
    ]
    
    # If the last chunk is shorter than 512, pad it
    if len(chunked_input_ids[-1]) < 512:
        chunked_input_ids[-1] = tokenizer.pad(
            {"input_ids": chunked_input_ids[-1]},
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )["input_ids"].squeeze().tolist()
    
    # Create labels as a copy for next-token prediction
    chunked_labels = chunked_input_ids.copy()

    return {
        'input_ids': chunked_input_ids,
        'labels': chunked_labels
    }
    

# Load the dataset in streaming mode
dataset = load_dataset('json', data_files=data_files, streaming=True)

def concatenate_and_tokenize_streaming(dataset):
    # This function will return an iterable batch of tokenized sequences
    def generator():
        # List to store tokenized sequences
        buffer_input_ids = []
        
        for example in dataset:
            # Tokenize each individual text
            tokenized = tokenizer(example["text"], truncation=False, return_tensors="pt")
            input_ids = tokenized["input_ids"].squeeze().tolist()
            
            # Add the tokenized input_ids and EOS token
            buffer_input_ids.extend(input_ids)
            buffer_input_ids.append(tokenizer.eos_token_id)
            
            # If buffer has enough tokens, yield chunked sequences of length 512
            while len(buffer_input_ids) >= 512:
                yield {
                    'input_ids': buffer_input_ids[:512],
                    'labels': buffer_input_ids[:512]  # Next token prediction
                }
                buffer_input_ids = buffer_input_ids[512:]
        
        # Handle the remainder (less than 512 tokens)
        if buffer_input_ids:
            # Pad the last sequence if it's shorter than 512 tokens
            padded_input_ids = tokenizer.pad(
                {"input_ids": buffer_input_ids},
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )["input_ids"].squeeze().tolist()
            
            yield {
                'input_ids': padded_input_ids,
                'labels': padded_input_ids  # Next token prediction
            }
    
    return IterableDataset.from_generator(generator)

# Apply the function to the entire dataset
if not streaming:
# Load dataset
    dataset = load_dataset('json', data_files=data_files)

    tokenized_data_train = concatenate_and_tokenize_all(dataset['train'])
    tokenized_data_val = concatenate_and_tokenize_all(dataset['validation'])

    tokenized_datasets = DatasetDict({
        'train': Dataset.from_dict(tokenized_data_train),
        'validation': Dataset.from_dict(tokenized_data_val)
    })

    print('Length training dataset', len(tokenized_datasets['train']))
    # print first example
    print('First train example:')
    print(tokenized_datasets['train'][0])
    #print decoded version
    print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))

    
else:
    dataset = load_dataset('json', data_files=data_files, streaming=True)
    tokenized_data_train = concatenate_and_tokenize_streaming(dataset['train'])
    tokenized_data_val = concatenate_and_tokenize_streaming(dataset['validation'])

    tokenized_datasets = DatasetDict({
        'train': tokenized_data_train,
        'validation': tokenized_data_val
    })

    print('First train example:')
    iter = iter(tokenized_datasets['train'])
    batch = next(iter)
    #print decoded version
    print(tokenizer.decode(batch['input_ids']))



# # Assert that all train samples are length 512
# for sample in tokenized_datasets['train']:
#     assert len(sample['input_ids']) == 512, f"Sample length is {len(sample['input_ids'])}, expected 512"


training_args = TrainingArguments(
    output_dir=f"/n/holyscratch01/barak_lab/Users/nabreu/gpt2-finetuned-sweep-{slurm_index}",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_accumulation_steps=eval_accumulation_steps,
    num_train_epochs=num_epochs,
    save_total_limit=2,
    warmup_steps=warmup_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    lr_scheduler_type='cosine',
    logging_dir=f"./logs/sweep-{slurm_index}",
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False
)

# Custom query evaluation function with batch inference
def custom_query_evaluation(model, tokenizer, query_dataset, batch_size=8):
    total = len(query_dataset)
    print('Evaluating query dataset with', total, 'samples')
    correct = 0

    tokenizer.padding_side = "left"

    # Split the queries into batches
    for i in range(0, total, batch_size):
        batch = query_dataset[i:i+batch_size]

        # Extract queries and correct answers for this batch
        queries = [example['query'].strip() for example in batch]
        correct_answers = [example['answers'] for example in batch]

        # Tokenize all queries in the batch
        inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        # Generate outputs for all queries in the batch
        outputs = model.generate(**inputs, max_new_tokens=20, num_return_sequences=1)

        # Decode the generated sequences
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        tails = [pred.strip().split('.')[0].replace(queries[j], '').strip() for j, pred in enumerate(generated_texts)]

        # Check each generated text against the correct answers for the batch
        for tail, correct_ans in zip(tails, correct_answers):
            if tail in correct_ans:
                correct += 1

        print(f"Batch {i} example:")
        print("Query:", queries[0])
        print("Correct answer:", correct_answers[0])
        print("Generated text:", generated_texts[0])
        print("Tail:", tails[0])
        print()

    tokenizer.padding_side = "right"
    accuracy = correct / total
    return {'eval/query_accuracy': accuracy}

# Trainer
trainer = CustomTrainer(
    cfg=config_dict,
    cloze_task_fn=custom_query_evaluation,
    eval_query_dataset_path=eval_query_dataset_path,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

trainer.log_config()

# Save the final model
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
