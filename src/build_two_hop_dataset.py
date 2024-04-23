from wikidata.utils import get_label, load_json
from build_benchmark import all_relevant_facts_given_list_of_subjects, sample_relevant_facts_given_list_of_subjects
from query import Query
from pathlib import Path
import json
import argparse
import os
import random
from query import Query
from pathlib import Path
from wikidata.utils import get_label, load_json
from build_one_hop_dataset import to_file

from wikidata.relations import our_relations, relation2impacted_relations, relation2phrase
from wikidata.utils import subject_relation_to_targets, ent_to_relation_ids, get_label, get_aliases, get_description, \
    subjects_given_relation_target
from build_logical_constraints import generate_constraints
from utils import create_test_example_given_input_targets
from relation import Relation
from query import Query, TwoHopQuery
from testcase import TestCase
from two_hop_phrases import relation_couple_to_phrase

def update_facts(facts, s, r, o):
    f_string = f'{str(s)}_{str(r.id())}_{str(o)}'
    if f_string not in facts:
        facts[f_string] = Query(s, r, o).to_dict()

def update_two_hop_facts(facts, s, r, o, r2, o2, phrase):
    f_string = f'{str(s)}_{str(r.id())}_{str(o)}_{str(r2.id())}_{str(o2)}'
    if f_string not in facts:
        facts[f_string] = TwoHopQuery(s, r, o, r2, o2, phrase).to_dict()

def forward_two_hop_axis(subject_id: str, relation: Relation, target_id: str, facts: dict, two_hop_facts: dict):
    tests = []
    if not target_id or target_id[0] != 'Q':
        return tests
    for backward_relation in Relation:
        backward_relation_id = backward_relation.id()
        backward_subjects = subjects_given_relation_target(backward_relation_id, subject_id)
        for backward_subject in backward_subjects:
            phrase = relation_couple_to_phrase(backward_relation, relation)
            if phrase is None:
                continue
            phrase = phrase.replace('<subject>', get_label(backward_subject))
            update_facts(facts, backward_subject, backward_relation, subject_id)
            update_facts(facts, subject_id, relation, target_id)
            update_two_hop_facts(two_hop_facts, backward_subject, backward_relation, subject_id, relation, target_id, phrase)
    return tests

def two_hop_axis(subject_id: str, relation: Relation, target_id: str, facts: dict, two_hop_facts: dict):
    tests = []
    if not target_id or target_id[0] != 'Q':
        return tests
    target_relations = ent_to_relation_ids(target_id)
    for relation_id in target_relations:
        second_relation_enum = Relation.id_to_enum(relation_id)
        if second_relation_enum is None:
            continue
        second_hop_targets = subject_relation_to_targets(target_id, second_relation_enum)
        for second_hop_target in second_hop_targets:
            phrase = relation_couple_to_phrase(relation, second_relation_enum)
            if phrase is None:
                continue
            phrase = phrase.replace('<subject>', get_label(subject_id))
            test_query = TwoHopQuery(subject_id, relation, target_id, second_relation_enum, second_hop_target, phrase)
            update_facts(facts, target_id, second_relation_enum, second_hop_target)
            update_facts(facts, subject_id, relation, target_id)
            update_two_hop_facts(two_hop_facts, subject_id, relation, target_id, second_relation_enum, second_hop_target, phrase)
    return tests


def build_two_hop_dataset(args):
    random.seed(args.seed)
    DATA_DIR = '/Users/annabelle/workplace/RippleEdits/data/benchmark'
    fname = os.path.join(DATA_DIR, args.facts_file)
    facts = load_json(fname)
    facts_str = list(facts.keys())
    random.shuffle(facts_str)

    one_hop_facts = {}
    two_hop_facts = {}
    for i, f in enumerate(facts_str):
        query = facts[f]
        q = Query.from_dict(query)
        print(f'At fact {i}')
        for target in q._targets_ids:
            two_hop = two_hop_axis(q._subject_id, q._relation, target, one_hop_facts, two_hop_facts)
            forward_two_hop = forward_two_hop_axis(q._subject_id, q._relation, target, one_hop_facts, two_hop_facts)
            print(f'Length two hop: {len(two_hop_facts.keys())}')
            print(f'Length one hop: {len(one_hop_facts.keys())}')

            if len(two_hop_facts.keys()) > args.num_facts:
                break
        else:
            continue
        break

    two_hop_fname = f'two_hop_fom_{args.facts_file}'
    two_hop_fname_full = os.path.join(DATA_DIR, two_hop_fname)
    one_hop_fname = f'one_hop_from_{two_hop_fname}'
    one_hop_fname = os.path.join(DATA_DIR, one_hop_fname)
    to_file(two_hop_fname_full, two_hop_facts)
    to_file(one_hop_fname, one_hop_facts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_facts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--facts_file", type=str, default='top_ents_5_entities_0_facts_each.json')
    build_two_hop_dataset(parser.parse_args())