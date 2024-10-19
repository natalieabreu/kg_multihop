import sys, os
sys.path.append('..')

import json
import pickle
import random
import itertools
from datetime  import datetime

from kg.kg import KG as KG_old
from kg.kg2 import KG
from kg.dataset_generator import DatasetGenerator

def load_from_pkl(path='/n/home07/nabreu/kg_multihop/kg/data/kg_popular.pkl'):
    with open(path, 'rb') as f:
        kg = pickle.load(f)

    print ('# Entities', len(kg.get_all_entities()))
    print ('# Relations', len(kg.relation_types))
    print('# Edges', len(kg.edges))

    return kg

def save_to_pkl(kg, path):
    with open(path, 'wb') as f:
        pickle.dump(kg, f)

def generate_eval_dataset_nodes(kg, data_outfile):
    samples = []
    for ent in kg.entities:
        sample = kg.get_entity_paragraph(ent)
        samples.append(sample)
    
    # shuffle samples
    random.shuffle(samples)

    with open(data_outfile, 'w') as f:
        json.dump({'samples': samples}, f)

def generate_eval_dataset_edges(kg, data_outfile):
    samples = []
    for edge in kg.edges:
        sentence = kg.get_sentence(edge)
        samples.append(sentence)
    
    # shuffle samples
    random.shuffle(samples)

    with open(data_outfile, 'w') as f:
        json.dump({'samples': samples}, f)

def generate_eval_dataset_queries(kg, data_outfile):
    samples = []
    for edge in kg.edges:
        samples.append(kg.get_labeled_query(edge))

    unique_queries = []
    seen = set()
    for sample in samples:
        if sample['query'] not in seen:
            unique_queries.append({'query': sample['query'], 'answers': [sample['answer']]})
            seen.add(sample['query'])
        else:
            for q in unique_queries:
                if q['query'] == sample['query']:
                    q['answers'].append(sample['answer'])

    print('Queries', len(samples))
    print('Unique queries:', len(unique_queries))
    
    # shuffle samples
    random.shuffle(unique_queries)

    with open(data_outfile, 'w') as f:
        json.dump({'samples': unique_queries}, f)

def build_generator_and_dataset(kg,
                    data_outfile,
                    generator_outfile,
                    num_experts,
                    expert_strategy,
                    num_samples,
                    entity_sampling_strategy,
                    paragraph_strategy='entity',
                    **kwargs):
    
    # if not os.path.exists(outpath):
    #     os.makedirs(outpath)

    print("#########")
    print("Building generator and dataset")
    print("\tExpert strategy:", expert_strategy)
    print("\tNumber of experts:", num_experts)
    print("\tNumber of samples:", num_samples)
    print("\tEntity sampling strategy:", entity_sampling_strategy)
    print("\tParagraph strategy:", paragraph_strategy)
    print("#########")

    clusters = None
    if expert_strategy == 'denoising' or expert_strategy == 'skill_selection':
        with open('/n/home07/nabreu/kg_multihop/kg/data/spectral_clustering_clusters_cluster_qr_2500.json', 'r') as f:
            clusters = json.load(f)

    generator = DatasetGenerator(kg, clusters=clusters)
    generator.generate_experts(
        expert_strategy, 
        num_experts=num_experts,
        **kwargs)
    generator.save_generator_to_pkl(f"{generator_outfile}.pkl", {'expert_strategy': expert_strategy, 'num_experts': num_experts, 'kwargs': kwargs})
    generator.generate_and_save_dataset(
        num_samples, 
        entity_sampling_strategy, 
        paragraph_strategy,
        data_outfile
    )

def generate_dataset_from_generator(
        generator_path, 
        data_outfile,
        num_samples,
        entity_sampling_strategy,
        paragraph_strategy='entity'
    ):
    with open(generator_path, 'rb') as f:
        obj = pickle.load(f)

    generator = obj['generator']
    config = obj['expert_args']

    print('Generating dataset from generator with expert config:', config)
    
    generator.generate_and_save_dataset(
        num_samples, 
        entity_sampling_strategy, 
        paragraph_strategy,
        data_outfile
    )


if __name__ == '__main__':
    kg = load_from_pkl('/n/home07/nabreu/kg_multihop/kg/data/ground_truth.pkl')

    # kg = load_from_pkl('/n/home07/nabreu/kg_multihop/kg/data/kg_orig_entities.pkl')
    # suffix = 'orig_entities'
    suffix=''
    suffix = '_' + suffix if suffix else ''

    # generate_eval_dataset_edges(kg, f'/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset{suffix}.json')
    # generate_eval_dataset_queries(kg, f'/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries{suffix}.json')

    # # kg.set_relation_templates('random+all')

    # # generate_eval_dataset_queries(kg, '/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/eval_dataset_queries.json')

    # print('Done')
    # sys.exit()
    n_experts = [1000, 5000, 10000]
    conf = [0.4, 0.5, 0.6, 0.7, 0.9]
    for e,c in itertools.product(n_experts, conf):
        args = {
            'num_experts': e,
            'expert_strategy': 'denoising',
            'num_samples': 1_000_000,
            'entity_sampling_strategy': 'uniform',
            'entities_per_expert': len(kg.get_all_entities()),
            'confidence': c,
            'paragraph_strategy':'entity'
        }

        # args = {
        #     'num_experts': 1_000,
        #     'expert_strategy': 'random_walk',
        #     'num_samples': 500_000,
        #     'entity_sampling_strategy': 'uniform',
        #     'facts_per_expert': 10_000,
        # }

        eventid = datetime.now().strftime('%Y%m-%d%H-%M%S')

        data_outpath = f"/n/holyscratch01/barak_lab/Users/nabreu/transcendence_datasets/{args['expert_strategy']}_{args['num_samples']}{suffix}"

        if args['expert_strategy'] in ['denoising', 'skill_selection']:
            generator_outpath = f"/n/holyscratch01/barak_lab/Users/nabreu/transcendence_generators/{args['expert_strategy']}_nexperts{args['num_experts']}_N{args['entities_per_expert']}_c{args['confidence']}{suffix}"
        elif args['expert_strategy'] == 'random_walk':
            generator_outpath = f"/n/holyscratch01/barak_lab/Users/nabreu/transcendence_generators/{args['expert_strategy']}_nexperts{args['num_experts']}_M{args['facts_per_expert']}{suffix}"
        else:
            raise ValueError('Invalid expert strategy')
        
        if not os.path.exists(data_outpath):
            os.makedirs(data_outpath)
        if not os.path.exists(generator_outpath):
            os.makedirs(generator_outpath)

        data_outfile = f"{data_outpath}/{eventid}.json"
        generator_outfile = f"{generator_outpath}/{eventid}"
        with open(data_outfile, 'w') as f:
            json.dump({'config': args}, f)

        build_generator_and_dataset(kg=kg, data_outfile=data_outfile, generator_outfile=generator_outfile, **args)
        
        data_files_path = '/n/home07/nabreu/kg_multihop/train/data_files.json'
        with open(data_files_path, 'r') as f:
            data_files = json.load(f)

        # add new data file to dict data_files
        data_files[eventid] = { **args, 'filepath':data_outfile, 'notes': suffix.strip('_') }

        with open(data_files_path, 'w') as f:
            json.dump(data_files, f)

        
        # generate_dataset_from_generator(
        #     '/n/holyscratch01/barak_lab/Users/nabreu/transcendence_generators/denoising_nexperts1000_N25803_c0.8/202408-2016-0733.pkl', 
        #     data_outfile, args['num_samples'], args['entity_sampling_strategy'], paragraph_strategy='entity_out')
        print('Saved to', data_outfile)
        print('Done')
        