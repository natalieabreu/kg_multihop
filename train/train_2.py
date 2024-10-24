import os, sys
import itertools
import yaml

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import numpy as np
import json
from datasets import load_dataset, IterableDataset
import torch.distributed as dist

sys.path.append("..")
from eval.cloze_task import cloze_task_evaluation

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
            return get_samples_file(config_or_label), DATA_FILE_MAP[config_or_label]
        else:
            raise ValueError(f"No matching label '{config_or_label}' found in mappings.")

    # Otherwise, assume it's a dictionary of attributes
    elif isinstance(config_or_label, dict):
        for lbl, entry in DATA_FILE_MAP.items():
            if all(config_or_label[key] == entry[key] for key in config_or_label):
                return get_samples_file(lbl), entry
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
        
        dist.barrier()

    return samples_file



def main():
    dist.init_process_group("nccl")
    dist.barrier()
    rank = dist.get_rank()

        # Load config file
    with open("/n/home07/nabreu/kg_multihop/train/train_2_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Get SLURM array task ID (job index)
    slurm_index = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    SLURM_ID = os.getenv("SLURM_JOB_ID", "0")

    # Create a list of all possible hyperparameter combinations


    # train_path = config.get("train_path", "/n/home07/nabreu/kg_multihop/202409-1123-4731-samples.json") # 1M p=1
    

    hyperparams = config.get("hyperparams", {})
    per_device_train_batch_size = hyperparams.get("per_device_train_batch_size", 8)
    per_device_eval_batch_size = hyperparams.get("per_device_eval_batch_size", 2)
    eval_accumulation_steps = hyperparams.get("eval_accumulation_steps", 100)

    def ensure_list(val):
        return val if isinstance(val, list) else [val]

    # Ensure hyperparameters are lists, if not, make them lists
    num_epochs = ensure_list(hyperparams.get("num_epochs", 100))
    max_samples = ensure_list(hyperparams.get("max_samples", None))
    learning_rate = ensure_list(hyperparams.get("learning_rate", 0.001))
    weight_decay = ensure_list(hyperparams.get("weight_decay", 0.1))
    gradient_accumulation_steps = ensure_list(hyperparams.get("gradient_accumulation_steps", 3))
    warmup_steps = ensure_list(hyperparams.get("warmup_steps", 1000))

    # Handle train_data_config separately if it needs to be included in the sweep
    train_data_config = config.get("train_data_config", "config_p1.0_1000experts_1M")
    train_data_config = ensure_list(train_data_config)

    # Create list of all combinations for only those parameters that are lists
    param_combinations = list(itertools.product(
        num_epochs,
        max_samples,
        learning_rate,
        weight_decay,
        gradient_accumulation_steps,
        warmup_steps,
        train_data_config
    ))

    # Select the combination for this job based on SLURM_ARRAY_TASK_ID
    chosen_combination = param_combinations[slurm_index]

    # Unpack the chosen combination into individual hyperparameters
    num_epochs, max_samples, learning_rate, weight_decay, gradient_accumulation_steps, warmup_steps, train_data_config = chosen_combination

    # Get the file path based on the data config if it exists
    train_path, data_config = get_filepath(train_data_config)
    eval_path = config.get("eval_path", "/n/home07/nabreu/kg_multihop/eval-samples.json")
    eval_query_dataset_path = config.get("eval_query_dataset_path", "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json")
    streaming = config.get("streaming", False)
    notes = config.get("notes", "")

    data_files = {
        'train': train_path,
        'validation': eval_path
    }

    config_dict = {
        "num_epochs": num_epochs,
        "max_samples": max_samples,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "train_path": train_path,
        "eval_path": eval_path,
        "eval_query_dataset_path": eval_query_dataset_path,
        "notes": notes,
        "run_id": SLURM_ID,
        "train_data_config": dict(data_config),
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
    def concatenate_and_tokenize_all(dataset, max_samples=None):
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

            if max_samples and count >= max_samples:
                print(f'Using first {count} samples')
                break
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
        

    def concatenate_and_tokenize_streaming(dataset, max_samples=None):
        # This function will return an iterable batch of tokenized sequences
        def generator():
            # List to store tokenized sequences
            buffer_input_ids = []
            
            count = 0
            for example in dataset:
                # Tokenize each individual text
                tokenized = tokenizer(example["text"], truncation=False, return_tensors="pt")
                input_ids = tokenized["input_ids"].squeeze().tolist()
                
                # Add the tokenized input_ids and EOS token
                buffer_input_ids.extend(input_ids)
                buffer_input_ids.append(tokenizer.eos_token_id)
                count += 1

                # If buffer has enough tokens, yield chunked sequences of length 512
                while len(buffer_input_ids) >= 512:
                    yield {
                        'input_ids': buffer_input_ids[:512],
                        'labels': buffer_input_ids[:512]  # Next token prediction
                    }
                    buffer_input_ids = buffer_input_ids[512:]

                if max_samples and count >= max_samples:
                    print(f'Using first {count} samples')
                    break
            
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

        if rank == 0:
            print('Length training dataset', len(tokenized_datasets['train']))
            # print first example
            print('First train example:')
            print(tokenized_datasets['train'][0])
            #print decoded version
            print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))

        dist.barrier()

        
    else:
        dataset = load_dataset('json', data_files=data_files, streaming=True)
        tokenized_data_train = concatenate_and_tokenize_streaming(dataset['train'])
        tokenized_data_val = concatenate_and_tokenize_streaming(dataset['validation'])

        tokenized_datasets = DatasetDict({
            'train': tokenized_data_train,
            'validation': tokenized_data_val
        })

        if rank == 0:
            print('First train example:')
            iter = iter(tokenized_datasets['train'])
            batch = next(iter)
            #print decoded version
            print(tokenizer.decode(batch['input_ids']))

        dist.barrier()



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

    

    # Trainer
    trainer = CustomTrainer(
        cfg=config_dict,
        cloze_task_fn=cloze_task_evaluation,
        eval_query_dataset_path=eval_query_dataset_path,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()


    # Save the final model
    
    model.save_pretrained(f"/n/holyscratch01/barak_lab/Users/nabreu/transcendence_models/gpt2-finetuned/{SLURM_ID}")
    # tokenizer.save_pretrained("./gpt2-finetuned")

    dist.destroy_process_group()
    print(f'Done {rank}')


if __name__ == "__main__":
    main()