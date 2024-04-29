from wikidata.utils import load_json
import glob
from query import Query

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

for fname in glob.glob('/n/holyscratch01/kempner_lab/Everyone/data/twohop/top_ents_5000_entities_0_facts_each_5000_facts/one_hop*'):
    # Load facts as dicts
    one_hop.update(load_json(fname))

    # Optional: load facts as Query objects
    # partial = {}
    # dicts = load_json(fname)
    # for uuid in dicts.keys():
    #     partial[uuid] = Query.from_dict(dicts[uuid])
    # one_hop.update(partial)