import torch
import wandb
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import concurrent.futures
import torch.nn.functional as F


from train import Config

class ModelEvaluator:
    def __init__(self, model_name, checkpoint_path, device='cuda'):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(model_name, vocab_size=len(self.tokenizer), ignore_mismatched_sizes=True)
        
        if model_name == 'meta-llama/Meta-Llama-3-8B':
            # LoRA configuration
            lora_config = LoraConfig(
                    r=8,                       # Rank of the low-rank matrix
                    lora_alpha=32,             # Scaling factor for the low-rank matrices
                    target_modules=["q_proj", "v_proj"],  # Target layers for LoRA
                    lora_dropout=0.1,          # Dropout for LoRA layers
                    bias="none"                # Whether to add a bias term
                )

                # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
        self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")

    def to_device(self, inputs):
        return inputs.to(self.device)
    
    def calculate_first_token_logit_prob(self, query, tail):
        input_ids = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)

            logits = outputs.logits[:, -1, :]  # Get logits for the first new token

            # The first new token is the first token in the prediction
            first_new_token_id = self.tokenizer.encode(tail, return_tensors='pt')[0, 0].item()

            log_probs = F.log_softmax(logits, dim=-1)
            first_token_log_prob = log_probs[0, first_new_token_id].item()

            highest_prob_token_id = torch.argmax(logits).item()

        return first_token_log_prob, highest_prob_token_id
    
    def evaluate_probability(self, dataset, subset=True):
        if subset:
            return self.evaluate_probability_subset(dataset, num_queries=1000)

        self.model.eval()

        print(f'Evaluating probability on {len(dataset)} queries...')

        def process_item(item):
            query = item['query']
            correct_answers = item['answers']

            first_token_log_prob, highest_prob_token_id = self.calculate_first_token_logit_prob(query, correct_answers[0])
            return (query, correct_answers, first_token_log_prob, highest_prob_token_id)

        with torch.no_grad(), concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_item, dataset))

        for i, (query, correct_answers, first_token_log_prob, highest_prob_token_id) in enumerate(results):
            if i < 10:
                print(f'Query: {query}')
                print(f'Correct answers: {correct_answers}')
                print(f'First token log probability: {first_token_log_prob}')
                print(f'Highest probability token: {self.tokenizer.decode(highest_prob_token_id)}')

                print()

        self.model.train()
        avg = sum(first_token_log_prob for _, _, first_token_log_prob, _ in results) / len(results)
        return avg
    
    def evaluate_probability_subset(self, dataset, method='sampling', temperature=1.0, max_new_tokens=20, num_beams=5, num_queries=1000):
        self.model.eval()

        print(f'Evaluating probability on {num_queries} queries with method={method}, temperature={temperature}, max_new_tokens={max_new_tokens}, num_beams={num_beams}...')

        def process_item(item):
            query = item['query']
            correct_answers = item['answers']

            first_token_log_prob, highest_prob_token_id = self.calculate_first_token_logit_prob(query, correct_answers[0])
            return (query, correct_answers, first_token_log_prob, highest_prob_token_id)

        results = []
        for i in range(num_queries):
            item = dataset[i]
            res = process_item(item)
            results.append(res)
            query, correct_answers, first_token_log_prob, highest_prob_token_id = res

            if i < 10:
                print(f'Query: {query}')
                print(f'Correct answers: {correct_answers}')
                print(f'First token log probability: {first_token_log_prob}')
                print(f'Highest probability token: {self.tokenizer.decode(highest_prob_token_id)}')
                print()

        self.model.train()
        avg = sum(first_token_log_prob for _, _, first_token_log_prob, _ in results) / num_queries
        return avg

    def generate_answer(self, query, method='sampling', temperature=1.0, max_new_tokens=20, num_beams=5):
        inputs = self.tokenizer.encode(query, return_tensors='pt')
        inputs = self.to_device(inputs)
        
        if method == 'sampling':
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature
            )
        elif method == 'beam_search':
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                num_beams=num_beams,
                early_stopping=True
            )
        elif method == 'greedy':
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False
            )
        else:
            raise ValueError("Invalid generation method. Choose 'sampling' or 'beam_search'.")

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split('.')[0]

    def evaluate_accuracy(self, dataset, method='sampling', temperature=1.0, max_new_tokens=20, num_beams=5, subset=True):
        if subset:
            return self.evaluate_accuracy_subset(dataset, method, temperature, max_new_tokens, num_beams, num_queries=1000)

        correct_predictions = 0
        total_queries = len(dataset)
        self.model.eval()

        print(f'Evaluating accuracy on {total_queries} queries with method={method}, temperature={temperature}, max_new_tokens={max_new_tokens}, num_beams={num_beams}...')

        def process_item(item):
            query = item['query']
            correct_answers = item['answers']

            prediction = self.generate_answer(query, method, temperature, max_new_tokens, num_beams)
            tail = prediction.replace(query, "").strip()

            is_correct = tail in correct_answers
            return (query, correct_answers, prediction, tail, is_correct)

        with torch.no_grad(), concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_item, dataset))

        correct_predictions = sum(1 for _, _, _, _, is_correct in results if is_correct)

        for i, (query, correct_answers, prediction, tail, is_correct) in enumerate(results):
            if i < 10:
                print(f'Query: {query}')
                print(f'Correct answers: {correct_answers}')
                print(f'Prediction: {prediction}')
                print(f'Tail: {tail}')
                print()

        accuracy = correct_predictions / total_queries
        self.model.train()
        return accuracy

    def evaluate_accuracy_subset(self, dataset, method='sampling', temperature=1.0, max_new_tokens=20, num_beams=5, num_queries=1000):
        correct_predictions = 0
        total_queries = num_queries
        self.model.eval()

        print(f'Evaluating accuracy on {total_queries} queries with method={method}, temperature={temperature}, max_new_tokens={max_new_tokens}, num_beams={num_beams}...')

        def process_item(item):
            query = item['query']
            correct_answers = item['answers']

            prediction = self.generate_answer(query, method, temperature, max_new_tokens, num_beams)
            tail = prediction.replace(query, "").strip()

            is_correct = tail in correct_answers
            return (query, correct_answers, prediction, tail, is_correct)

        for i in range(num_queries):
            item = dataset[i]
            res = process_item(item)
            query, correct_answers, prediction, tail, is_correct = res
            if is_correct:
                correct_predictions += 1

            if i < 10:
                print(f'Query: {query}')
                print(f'Correct answers: {correct_answers}')
                print(f'Prediction: {prediction}')
                print(f'Tail: {tail}')
                print()

        accuracy = correct_predictions / total_queries
        self.model.train()
        return accuracy
    
def evaluate_accuracy_with_query_results(self, dataset, method='sampling', temperature=1.0, max_new_tokens=20, num_beams=5, subset=True):
    if subset:
        dataset = dataset[:1000]

    query_results = {}
    total_queries = len(dataset)
    self.model.eval()

    print(f'Evaluating accuracy on {total_queries} queries with method={method}, temperature={temperature}, max_new_tokens={max_new_tokens}, num_beams={num_beams}...')

    def process_item(item):
        query = item['query']
        correct_answers = item['answers']

        prediction = self.generate_answer(query, method, temperature, max_new_tokens, num_beams)
        tail = prediction.replace(query, "").strip()

        is_correct = tail in correct_answers
        return (query, is_correct)

    with torch.no_grad(), concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, dataset))

    for query, is_correct in results:
        query_results[query] = is_correct

    correct_predictions = sum(1 for is_correct in query_results.values() if is_correct)
    accuracy = correct_predictions / total_queries
    self.model.train()
    return accuracy, query_results

    
def plot_query_correctness(checkpoints, all_results):
    # prompt: for each query, if the query correctness_list has a negative slope at some point (the query was correct at an early checkpoint and incorrect at a later one), color it red in the plot

    import matplotlib.pyplot as plt

    # Assuming you have a list of tuples (checkpoint_path, accuracy, query_results)
    # in the variable 'all_results'

    # Create a dictionary to store the correctness of each query over checkpoints
    query_correctness = {}
    for checkpoint_path, accuracy, query_results in all_results:
        for query, is_correct in query_results.items():
            if query not in query_correctness:
                query_correctness[query] = []
            query_correctness[query].append(is_correct)

    # Plot the correctness of each query over checkpoints
    plt.figure(figsize=(15, 10))
    for query, correctness_list in query_correctness.items():
        has_negative_slope = False
        for i in range(len(correctness_list) - 1):
            if correctness_list[i] == True and correctness_list[i+1] == False:
                has_negative_slope = True
                break
        if has_negative_slope:
            plt.plot(range(len(checkpoints)), correctness_list, label=query, color='red')
        else:
            plt.plot(range(len(checkpoints)), correctness_list, label=query)

    plt.xlabel("Checkpoint")
    plt.ylabel("Correctness")
    plt.title("Correctness of Queries over Checkpoints")
    plt.legend(loc='best')
    plt.savefig('query_correctness.png')
    
def get_all_checkpoints(checkpoint_dir):
    # Get all subdirectories in the specified checkpoint directory
    return [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, d))]

# Example usage of the script:
if __name__ == "__main__":
    # Initialize wandb
    

    # Load your dataset here
    with open("/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json","r") as f:
        dataset = json.load(f)['samples']

    # Define model checkpoints to evaluate
    # model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = 'gpt2'

    checkpoint_dir = "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_models/denoising_1000000_outneighbors/202409-1123-4731/gentle-night-125/"
    checkpoints = get_all_checkpoints(checkpoint_dir)

    # keep the checkpoints that end with '9999.pt'
    checkpoints = [c for c in checkpoints if c.endswith('9999.pt')]

    # sort checkpoints by steps in the checkpoint name
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    print('Running eval on checkpoints:')
    print(checkpoints)

    # checkpoints = ['/n/holyscratch01/barak_lab/Users/nabreu/transcendence_models/denoising_1000000_outneighbors/202409-1123-4731/gentle-night-125/5555.pt']

    # Evaluate with different generation settings
    methods = ['sampling', 'beam_search']
    temperatures = [0.0001, 0.001, 0.01, 0.1]

    all_results = []

    for checkpoint in checkpoints:
        wandb.init(project="transcendence-model-evaluation", entity="natalieabreu")
        evaluator = ModelEvaluator(model_name, checkpoint)

        # avg_prob_correct_first_token = evaluator.evaluate_probability(dataset)

        # wandb.log({
        #     "checkpoint": checkpoint,
        #     "avg_prob_correct_first_token": avg_prob_correct_first_token
        # })

        # for method in methods:
        #     for temp in temperatures:
        #         accuracy = evaluator.evaluate_accuracy(dataset, method=method, temperature=temp, max_new_tokens=20, num_beams=5, subset=False)

        #         # Log the results to wandb
        #         wandb.log({
        #             "checkpoint": checkpoint,
        #             "method": method,
        #             "temperature": temp,
        #             f"method={method}-temp={temp}-accuracy": accuracy
        #         })

        #         print(f"Accuracy for checkpoint={checkpoint}, method={method}, temperature={temp}: {accuracy:.2f}")

        method = 'greedy'
        # accuracy = evaluator.evaluate_accuracy(dataset, method=method, temperature=None, max_new_tokens=20, num_beams=None, subset=False)
        accuracy, results = evaluate_accuracy_with_query_results(evaluator, dataset, method=method, temperature=None, max_new_tokens=20, num_beams=None, subset=True)
        all_results.append((checkpoint, accuracy, results))

        # Log the results to wandb
        wandb.log({
            "checkpoint": checkpoint,
            "method": method,
            "temperature": 0,
            f"method={method}-accuracy": accuracy
        })

        print(f"Accuracy for checkpoint={checkpoint}, method={method}, temperature={0}: {accuracy:.2f}")

    # End the wandb run
    wandb.finish()

    # Plot the correctness of queries over checkpoints
    plot_query_correctness(checkpoints, all_results)
