from wikidata.utils import load_json, write_json
import glob
from query import Query
import random

seed = 12345
random.seed(seed)

two_hop = {}
one_hop = {}
for fname in glob.glob('/n/holyscratch01/kempner_lab/Everyone/data/twohop/top_ents_5000_entities_0_facts_each_5000_facts/two_hop*'):
    # Load facts as dicts
    two_hop.update(load_json(fname))

    # Optional: load facts as Query objects
    # partial = {}
    # dicts = load_json(fname)
    # for uuid in dicts.keys():
    #     partial[uuid] = Query.from_dict(dicts[uuid])
    # two_hop.update(partial)

facts_str = list(two_hop.keys()) 
print(f'{len(facts_str)} two hop facts')
write_json(two_hop,'/n/holyscratch01/kempner_lab/Everyone/data/multihop/two_hop_dataset.json')

facts_str = facts_str[:41000]
train_facts = facts_str[:32800]
test_facts = facts_str[32800:]

train = {}
test = {}
for f in train_facts:
    train[f] = two_hop[f]

for f in test_facts:
    test[f] = two_hop[f]

print(f'{len(train_facts)} train facts')
print(f'{len(test_facts)} test facts')

write_json(train,'/n/holyscratch01/kempner_lab/Everyone/data/multihop/train_two_hop_dataset.json')
write_json(test,'/n/holyscratch01/kempner_lab/Everyone/data/multihop/test_two_hop_dataset.json')

for fname in glob.glob('/n/holyscratch01/kempner_lab/Everyone/data/twohop/top_ents_5000_entities_0_facts_each_5000_facts/one_hop*'):
    # Load facts as dicts
    one_hop.update(load_json(fname))

    # Optional: load facts as Query objects
    # partial = {}
    # dicts = load_json(fname)
    # for uuid in dicts.keys():
    #     partial[uuid] = Query.from_dict(dicts[uuid])
    # one_hop.update(partial)

print(f'{len(one_hop.keys())} one hop facts')
write_json(one_hop,'/n/holyscratch01/kempner_lab/Everyone/data/multihop/one_hop_dataset.json')