import argparse
from wikidata.utils import load_json
from queryexecutor import GPT2QueryExecutor, GPT3QueryExecutor, GPTJQueryExecutor, GPTNeoXQueryExecutor, \
    LlamaQueryExecutor
from query import Query

def evaluate_dataset(dataset, query_executor, num_eval=100, type='one_hop'):
    #TODO: add two hop and three hop evaluation
    total = len(dataset.keys())
    correct = 0
    for i, one_hop_fact in enumerate(dataset.keys()):
        if i > num_eval:
            break
        q = Query.from_dict(dataset[one_hop_fact])
        res = query_executor.execute_query(q)
        # Returns true if response in possible answers
        if res:
            correct +=1 
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
            dataset = datasets[dataset_name]
            total, correct = evaluate_dataset(dataset, query_executor, num_eval=args.num_eval, type=dataset_name)        
            data['Total'].append(total)
            data['Correct'].append(correct)
            data['Accuracy'].append(correct/total)

    


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eval", type=int, default=100)
    parser.add_argument("--onehop", type=str, default='/Users/annabelle/workplace/RippleEdits/src/one_hop_from_two_hop_fom_top_ents_5_entities_0_facts_each.json')
    #TODO: add two and three hop evaluation
    evaluation(parser.parse_args())