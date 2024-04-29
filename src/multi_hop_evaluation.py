import argparse
from wikidata.utils import load_json, write_json
from queryexecutor import GPT2QueryExecutor, GPT3QueryExecutor, GPTJQueryExecutor, GPTNeoXQueryExecutor, \
    LlamaQueryExecutor
from query import Query
from build_one_hop_dataset import to_file
import os

def evaluate_dataset(dataset, query_executor, num_eval=100, fact_type='one_hop'):
    #TODO: add two hop and three hop evaluation
    #TODO: add more sophisticated sampling for eval
    total = num_eval
    correct = 0
    for i, one_hop_fact in enumerate(dataset.keys()):
        if i > num_eval:
            break
        q = Query.from_dict(dataset[one_hop_fact])
        res = query_executor.execute_query(q)
        # Returns true if response in possible answers
        if res:
            correct +=1
    print(f'{fact_type} total: {total} correct: {correct} acc: {correct/total}')
    return total, correct

def evaluation(args):
    data = {'Fact Type': [], 'Total': [], 'Correct':[],'Accuracy': [], 'Model':[]}

    #TODO: add more models
    models = ['llama']

    one_hop_dataset = load_json(args.onehop)
    datasets = {'one_hop': one_hop_dataset}

    for model in models:
        if model == 'llama':
            query_executor = LlamaQueryExecutor()

        for dataset_name in datasets.keys():
            data['Fact Type'].append(dataset_name)
            data['Model'].append(model)
            dataset = datasets[dataset_name]
            total, correct = evaluate_dataset(dataset, query_executor, num_eval=args.num_eval, fact_type=dataset_name)        
            data['Total'].append(total)
            data['Correct'].append(correct)
            data['Accuracy'].append(correct/total)

    #TODO update naming
    DATA_DIR = '/n/holyscratch01/kempner_lab/Everyone/data/twohop'
    fname = os.path.join(DATA_DIR, f'llama_one_hop_{args.num_eval}.json')
    to_file(fname, data)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eval", type=int, default=100)
    parser.add_argument("--onehop", type=str, default='/n/holyscratch01/kempner_lab/Everyone/data/twohop/top_ents_5000_entities_0_facts_each_10_facts/one_hop_from_two_hop_fom_top_ents_5000_entities_0_facts_each_10_facts_12345.json')
    #TODO: add two and three hop evaluation
    evaluation(parser.parse_args())