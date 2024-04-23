from relation import Relation
from wikidata.utils import get_label
from query import Query
from fact import Fact

class TwoHopFact(Fact):

    def __init__(self, subject_id, relation1, bridge_id, relation2, target_id):
        super.__init(subject_id, relation1, target_id)
        self._relation2 = relation2
        self._bridge_id = bridge_id

    def get_bridge_label(self):
        return get_label(self._bridge_id)

    def get_relation2_label(self):
        return self._relation2.name.replace('_', ' ')

    def to_dict(self):
        return {
            'prompt': self.get_fact_phrased(),
            'subject_id': self._subject_id,
            'relation': self._relation.name,
            'bridge_id': self._bridge_id,
            'relation2': self._relation2.name,
            'target_id': self._target_id
        }

    @staticmethod
    def from_dict(d):
        return TwoHopFact(d['subject_id'], Relation[d['relation1']], d['bridge_id'], Relation(d['relation2']),d['target_id'])

    def __str__(self):
        return f'({self.get_subject_label()}, {self.get_relation_label()}, {self.get_target_label()})'

