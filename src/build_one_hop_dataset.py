from wikidata.utils import get_label, load_json
from build_benchmark import all_relevant_facts_given_list_of_subjects, sample_relevant_facts_given_list_of_subjects
from query import Query
from pathlib import Path
import json
import argparse
import os
import random

def to_file(filename, data):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def build_one_hop_dataset(args):
    random.seed(args.seed)
    # ent_label2id_dict = load_json('./wikidata/ent_label2id/ent_label2id.json')
    
    # # Use entity IDs to get facts, not labels
    # ents = list(ent_label2id_dict.values())
    ent_fname = '/n/home01/nsaphra/workplace/twohop-1/src/top_entities_by_views_monthly.json'
    ent_dicts = load_json(ent_fname)
    ents = []
    for date in ent_dicts.keys():
        if date.startswith('2020') or date.startswith('2021'):
            ents = ents + [ent_dict['id'] for ent_dict in ent_dicts[date]]
    
    # Use entity IDs to get facts, not labels
    # ents = list(ent_label2id_dict.values())
    if args.num_ents:
            ents = random.sample(ents, args.num_ents)

    if args.num_facts:
        facts = sample_relevant_facts_given_list_of_subjects(ents, number_of_facts_each=args.num_facts)
    else:
        facts = all_relevant_facts_given_list_of_subjects(ents)
    
    queries = {}
    for i, f in enumerate(facts):
        f_string = f'{str(f[0])}_{str(f[1].id())}_{str(f[2])}'
        queries[f_string] = Query(f[0],f[1],f[2]).to_dict()

        if i % 100 == 0:
            print(f'At fact {i} of {len(facts)} facts')
    

    if args.num_ents or args.num_facts:
        filename = f'top_ents_{args.num_ents}_entities_{args.num_facts}_facts_each.json'
    else:
        filename = f'top_ents_all_entities_facts.json'

    filename = os.path.join(args.location,filename)
    to_file(filename, queries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ents", type=int, default=0)
    parser.add_argument("--num_facts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--location", type=str, default='/Users/annabelle/workplace/RippleEdits/data/benchmark')
    build_one_hop_dataset(parser.parse_args())