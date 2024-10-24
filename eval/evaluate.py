import wandb
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json

from safetensors.torch import load_file


from eval.cloze_task import cloze_task_evaluation

# Initialize wandb API
api = wandb.Api()

# Set up your wandb project and entity
WANDB_PROJECT = "transcendence-denoising"
WANDB_ENTITY = "natalieabreu"

# Path to the evaluation query dataset
EVAL_QUERY_DATASET_PATH = "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json"
EVAL_RESULTS_DIR = "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_eval_results/"
EVAL_PLOTS_DIR = "/n/home07/nabreu/kg_multihop/eval/plots/"

# Directory where your models are stored
MODEL_NAME = 'gpt2'
MODEL_DIR = "/n/holyscratch01/barak_lab/Users/nabreu/transcendence_models/gpt2-finetuned/"

class Evaluator:
    def __init__(self, runs, device='cuda'):
        self.device = device
        with open(EVAL_QUERY_DATASET_PATH, 'r') as f:
            self.eval_query_dataset = json.load(f)['samples']


        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.runs = runs

        # Create results directory if it doesn't exist
        if not os.path.exists(EVAL_RESULTS_DIR):
            os.makedirs(EVAL_RESULTS_DIR)

    def save_evaluation_data(self, run_id, data):
        """Save evaluation data to a JSON file."""
        eval_file = os.path.join(EVAL_RESULTS_DIR, f"{run_id}_eval.json")
        with open(eval_file, 'w') as f:
            json.dump(data, f, indent=4)

    def update_evaluation_data(self, run_id, data):
        """Update evaluation data in a JSON file."""
        eval_file = os.path.join(EVAL_RESULTS_DIR, f"{run_id}_eval.json")
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
            eval_data.update(data)
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=4)

    def load_evaluation_data(self, run_id):
        """Load evaluation data from a JSON file if it exists."""
        eval_file = os.path.join(EVAL_RESULTS_DIR, f"{run_id}_eval.json")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                return json.load(f)
        return None

    # Plot all runs' confidence vs accuracy
    def plot_confidence_vs_accuracy(self, runs=None):
        if runs is None:
            runs = self.runs

        # Dictionary to store confidence and accuracy values for each number of experts
        experts_dict = {}

        for run in runs:
            run_id = run.config['run_id']
            data_config = run.config['train_data_config']
            num_experts = data_config['num_experts']  # Extract the number of experts
            confidence = data_config['confidence']  # Extract confidence

            # Check if evaluation data already exists for this run
            eval_data = self.load_evaluation_data(run_id)
            if eval_data is None:
                eval_data = {}

            temperature_accuracies = eval_data.get('temperature_accuracies', {})
            
            temp = 0  # Using no sampling for this plot
            if str(temp) not in temperature_accuracies:
                model = self.load_model(run_id)
                print(f"Running evaluation for temperature: {temp} (run {run_id})")
                accuracy_result = cloze_task_evaluation(model, self.tokenizer, self.eval_query_dataset, temperature=temp)
                temperature_accuracies[str(temp)] = accuracy_result['eval/query_accuracy']
                eval_data['temperature_accuracies'] = temperature_accuracies
                self.save_evaluation_data(run_id, eval_data)
                accuracy = accuracy_result['eval/query_accuracy']
            else:
                print(f"Skipping evaluation for temperature: {temp} (already exists for run {run_id})")
                accuracy = temperature_accuracies[str(temp)]

            # Add the confidence and accuracy to the appropriate expert group
            if num_experts not in experts_dict:
                experts_dict[num_experts] = {'confidence': [], 'accuracy': []}
            
            experts_dict[num_experts]['confidence'].append(confidence)
            experts_dict[num_experts]['accuracy'].append(accuracy)

        # Plot one line for each number of experts, with sorted confidence
        plt.figure()
        for num_experts, values in experts_dict.items():
            # Sort confidence and accuracy based on confidence values
            sorted_conf_acc_pairs = sorted(zip(values['confidence'], values['accuracy']))
            sorted_confidence, sorted_accuracy = zip(*sorted_conf_acc_pairs)

            plt.plot(sorted_confidence, sorted_accuracy, marker='o', label=f"{num_experts} Experts")

        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Confidence vs Accuracy for Different Numbers of Experts")
        plt.legend()
        plt.savefig(os.path.join(EVAL_PLOTS_DIR, "confidence_vs_accuracy.png"))


    def plot_temperature_vs_accuracy(self, runs=None, temperatures=[0.0001, 0.001, 0.1, 1.0, 1.5]):
        """Plot sampling temperature vs accuracy for each run."""
        if runs is None:
            runs = self.runs
        sampling_temperatures = temperatures
        
        for run in runs:
            run_id = run.config['run_id']
            model = self.load_model(run_id)
            data_config = run.config['train_data_config']
            label = f"Confidence: {data_config['confidence']}, # Experts: {data_config['num_experts']}"
            
            # Check if evaluation data already exists for this run
            eval_data = self.load_evaluation_data(run_id)
            if eval_data is None:
                eval_data = {}

            temperature_accuracies = eval_data.get('temperature_accuracies', {})
            
            # Evaluate only for missing temperatures
            for temp in sampling_temperatures:
                if str(temp) not in temperature_accuracies:
                    print(f"Running evaluation for temperature: {temp} (run {run_id})")
                    accuracy_result = cloze_task_evaluation(model, self.tokenizer, self.eval_query_dataset, temperature=temp)
                    temperature_accuracies[str(temp)] = accuracy_result['eval/query_accuracy']
                else:
                    print(f"Skipping evaluation for temperature: {temp} (already exists for run {run_id})")

            # Save the updated evaluation data
            eval_data['temperature_accuracies'] = temperature_accuracies
            self.save_evaluation_data(run_id, eval_data)

            # Plot temperature vs accuracy for this run
            temperatures_sorted = sorted([float(t) for t in temperature_accuracies.keys()])
            accuracies_sorted = [temperature_accuracies[str(t)] for t in temperatures_sorted]
            plt.plot(temperatures_sorted, accuracies_sorted, marker='o', label=label)

        plt.xlabel("Sampling Temperature")
        plt.ylabel("Accuracy")
        plt.title("Sampling Temperature vs Accuracy for each Confidence")
        plt.legend()
        plt.savefig(os.path.join(EVAL_PLOTS_DIR, "temperature_vs_accuracy.png"))

    # Plot all runs' experts vs accuracy
    def plot_experts_vs_accuracy(self, runs=None):
        if runs is None:
            runs = self.runs

            # Dictionary to store num_experts and accuracy values for each confidence
            confidence_dict = {}

            for run in runs:
                run_id = run.config['run_id']
                data_config = run.config['train_data_config']
                num_experts = data_config['num_experts']  # Extract the number of experts
                confidence = data_config['confidence']  # Extract confidence

                # Check if evaluation data already exists for this run
                eval_data = self.load_evaluation_data(run_id)
                if eval_data is None:
                    eval_data = {}

                temperature_accuracies = eval_data.get('temperature_accuracies', {})
                
                temp = 0  # Using no sampling for this plot
                if str(temp) not in temperature_accuracies:
                    model = self.load_model(run_id)
                    print(f"Running evaluation for temperature: {temp} (run {run_id})")
                    accuracy_result = cloze_task_evaluation(model, self.tokenizer, self.eval_query_dataset, temperature=temp)
                    temperature_accuracies[str(temp)] = accuracy_result['eval/query_accuracy']
                    eval_data['temperature_accuracies'] = temperature_accuracies
                    self.save_evaluation_data(run_id, eval_data)
                    accuracy = accuracy_result['eval/query_accuracy']
                else:
                    print(f"Skipping evaluation for temperature: {temp} (already exists for run {run_id})")
                    accuracy = temperature_accuracies[str(temp)]

                # Add the num_experts and accuracy to the appropriate confidence group
                if confidence not in confidence_dict:
                    confidence_dict[confidence] = {'num_experts': [], 'accuracy': []}
                
                confidence_dict[confidence]['num_experts'].append(num_experts)
                confidence_dict[confidence]['accuracy'].append(accuracy)

            # Plot one line for each confidence, with sorted num_experts
            plt.figure()
            for confidence, values in confidence_dict.items():
                # Sort num_experts and accuracy based on num_experts values
                sorted_expert_acc_pairs = sorted(zip(values['num_experts'], values['accuracy']))
                sorted_num_experts, sorted_accuracy = zip(*sorted_expert_acc_pairs)

                plt.plot(sorted_num_experts, sorted_accuracy, marker='o', label=f"Confidence {confidence}")

            plt.xlabel("Number of Experts")
            plt.ylabel("Accuracy")
            plt.title("Experts vs Accuracy for Different Confidence Levels")
            plt.legend()
            plt.savefig(os.path.join(EVAL_PLOTS_DIR, "experts_vs_accuracy.png"))



    def load_model(self, run_id):
        """Load the model for a specific run"""
        model_path = os.path.join(MODEL_DIR, f"{run_id}/model.safetensors")
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, vocab_size=len(self.tokenizer))
        checkpoint = load_file(model_path, device=self.device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model from checkpoint: {model_path}")
        model.to(self.device)
        model.eval()
        return model
    
    # def load_checkpoint(self, checkpoint_path, model):
    #     checkpoint = load_file(checkpoint_path, device=self.device)
    #     # checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    #     model.load_state_dict(checkpoint)
    #     print(f"Loaded model from checkpoint: {checkpoint_path}")

    def evaluate_and_plot(self):
        self.plot_temperature_vs_accuracy()
        self.plot_confidence_vs_accuracy()
        self.plot_experts_vs_accuracy()

    def evaluate_model(self, run):
        """Evaluate a specific run."""
        run_id = run.id
        print(f"Evaluating run: {run_id}")

        # Evaluate confidence vs accuracy
        self.plot_confidence_vs_accuracy([run])

        # Evaluate sampling temperature vs accuracy
        self.plot_temperature_vs_accuracy([run])

        # Evaluate number of experts vs accuracy
        self.plot_experts_vs_accuracy([run])

def main():
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    evaluator = Evaluator(runs)
    evaluator.evaluate_and_plot()


if __name__ == "__main__":
    main()
