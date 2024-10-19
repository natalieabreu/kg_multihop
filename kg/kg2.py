import random
import pickle
import numpy as np 
from collections import deque, defaultdict

from kg.utils import *
from relation import Relation

from wikidata.utils import get_label

from kg.relation_templates import relations_templates, relation_id_to_name

# from sample_data import relation_templates

# class MockGPT4:
#     def __init__(self, *args, **kwargs):
#         pass

#     def get_relation_template(relation):
#         return relation_templates.get(relation, None)

class Entity:
    def __init__(self, id=None, name=None, aliases=None):
        if not id and not name:
            raise Exception('Must provide either id or name')
        
        if not name:
            name = get_label(id)

        self.id = id
        self.name = name
        self.in_degree = -1 
        self.out_degree = -1
        self.out_neighbors = None
        self.in_neighbors = None

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash((self.id, self.name))


class KG:
    def __init__(self, 
                 entities, # set of entities
                 relation_types, # set of relations
                 edges=None,
                 head=None,
                 tail=None,
                 relation=None,
                 relation_template_method='fixed'):
        
        if not edges and not head:
            raise ValueError("Either edges or head must be provided")
        
        self.entities = set(entities)
        self.id_to_entity = {
            entity.id: entity for entity in self.entities
        }
        self.relation_types = set(relation_types)
        

        if edges:
            self.set_edges(edges)

        else:
            self.set_hrt(head, relation, tail)

        self.set_all_degrees()

        self.set_relation_templates(method=relation_template_method)

    # def get_entity_type(self, entity):
    #     return self.entity_type_lookup[entity]
    
    def set_edges(self, edges):
        self.edges = list(set(edges))
        self.head = [edge[0] for edge in edges]
        self.relation = [edge[1] for edge in edges]
        self.tail = [edge[2] for edge in edges]

    def set_hrt(self, head, relation, tail):
        self.head = head
        self.relation = relation
        self.tail = tail
        self.edges = list(zip(head, relation, tail))        

    def get_all_entities(self):
        return self.id_to_entity.keys()

    def set_entity_name(self, entity, new_name):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]

        entity.name = new_name

    def get_out_neighbors(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
        return entity.out_neighbors

    def get_in_neighbors(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
        return entity.in_neighbors
    
    def sample_neighbor(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]

        # sample from both in and out neighbors proportional to number of neighbors in each set, 
        # returns ((rel, tail), 1) if out, ((head, rel), 0) if in
        if entity.out_degree == -1:
            self.set_out_degree(entity)

        if entity.in_degree == -1:
            self.set_in_degree(entity)

        if entity.out_degree == 0 and entity.in_degree == 0:
            return None
        
        if entity.out_degree == 0:
            return random.choice(entity.in_neighbors), 0
        
        if entity.in_degree == 0:
            return random.choice(entity.out_neighbors), 1

        if random.random() < len(entity.out_neighbors) / (len(entity.out_neighbors) + len(entity.in_neighbors)):
            return random.choice(entity.out_neighbors), 1
        
        return random.choice(entity.in_neighbors), 0

    '''
    Returns a dictionary mapping relation to list of entities that are the tail of that relation
    '''
    def get_tail_entities_by_relation(self):
        relation2entities = defaultdict(list)
        for _, r, t in self.edges:
            relation2entities[r].append(t)
        
        return relation2entities
    
    '''
    Returns a dictionary mapping relation to list of entities that are the head of that relation
    '''
    def get_head_entities_by_relation(self):
        relation2entities = defaultdict(list)
        for h, r, _ in self.edges:
            relation2entities[r].append(h)
        
        return relation2entities

    '''
    Returns number of connected components in graph
    '''
    def num_connected_components(self):
        graph = defaultdict(list)
        for h, r, t in self.edges:
            graph[h].append(t)
            graph[t].append(h)
        
        visited = set()
        num_components = 0
        for entity in self.entities:
            if entity.id not in visited:
                num_components += 1
                stack = [entity.id]
                while stack:
                    current_entity = stack.pop()
                    visited.add(current_entity)
                    for neighbor in graph[current_entity]:
                        if neighbor not in visited:
                            stack.append(neighbor)
        
        return num_components


            
    def random_walk(self, start_entity, k):
        if isinstance(start_entity, str):
            start_entity = self.id_to_entity[start_entity]

        current_entity = start_entity
        walk = [current_entity.id]
        
        for _ in range(k):
            # Find all possible next steps from the current entity
            if current_entity.out_degree == -1:
                current_entity.out_neighbors = [(self.relation[i], self.tail[i]) for i in range(len(self.head)) if self.head[i] == current_entity.id]
                current_entity.out_degree = len(current_entity.out_neighbors)
            # if current_entity not in self.entity_neighbors:
            #     self.entity_neighbors[current_entity] = [(self.relation[i], self.tail[i]) for i in range(len(self.head)) if self.head[i] == current_entity]
            next_steps = current_entity.out_neighbors
            # If there are no next steps, end the walk
            if not next_steps:
                break
            
            # Choose a random next step
            chosen_relation, next_entity = random.choice(next_steps)
            
            # Add the chosen relation and next entity to the walk
            walk.append(chosen_relation)
            walk.append(next_entity)

            # Update the current entity
            current_entity = self.id_to_entity[next_entity]
        
        return walk
    
    def random_walk_edge_set(self, start_entity, num_edges):
        if isinstance(start_entity, str):
            start_entity = self.id_to_entity[start_entity]

        edge_set = set()

        current_entity = start_entity

        while len(edge_set) < num_edges:
            (neighbor), direction = self.sample_neighbor(current_entity)

            if not neighbor:
                raise Exception('No neighbors found for node ' + current_entity.id)
            
            # ((rel, tail), 1) if out, ((head, rel), 0) if in
            if direction == 1:
                edge = (current_entity.id, neighbor[0], neighbor[1])
                current_entity = self.id_to_entity[neighbor[1]]
            else:
                edge = (neighbor[0], neighbor[1], current_entity.id)
                current_entity = self.id_to_entity[neighbor[0]]

            edge_set.add(edge)

        return list(edge_set)

            

    
    def shortest_path_lengths_within_k(self, start_id, head, relation, tail, k):
        # Create an adjacency list from the knowledge graph edges
        graph = defaultdict(list)
        for h, r, t in zip(head, relation, tail):
            graph[h].append((t, r))

        # BFS initialization
        queue = deque([(start_id, 0)])  # (current_entity, current_depth)
        visited = set()
        distances = {}

        while queue:
            current_entity_id, depth = queue.popleft()
            
            # If depth exceeds k, stop processing further
            if depth > k:
                continue
            
            # Store the distance to the current entity
            if current_entity_id not in visited:
                visited.add(current_entity_id)
                distances[current_entity_id] = depth
                
                # Enqueue all adjacent entities
                for neighbor, _ in graph[current_entity_id]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        
        return distances




    def save_subgraph(self, center, k):
        subentities = self.shortest_path_lengths_within_k(center, self.head, self.relation, self.tail, k).keys()
        entities = set([Entity(id=ent) for ent in subentities])
        subedges = [(h, r, t) for h, r, t in zip(self.head, self.relation, self.tail) if h in subentities and t in subentities]
        subrelation_types = set([r for _, r, _ in subedges])
        subgraph = KG(
            entities,
            subrelation_types,
            edges = subedges
        )

        center_str = '-'.join(center.split())

        with open(f'{center_str}_{k}.pkl', 'wb') as f:
            pickle.dump(subgraph, f)

    '''
    Remove edges of given relation type
    '''
    def filter_out_relation(self, relation_type):
        edges = [(h, r, t) for h, r, t in self.edges if r != relation_type]
        self.set_edges(edges)

        self.relation_types.remove(relation_type)
        self.set_all_degrees()

    '''
    Remove entity + edges involving entity
    '''
    # def remove_entity(self, entity):
    #     if isinstance(entity, Entity):
    #         entity_id = entity.id
    #     else:
    #         entity_id = entity

    #     edges = [(h, r, t) for h, r, t in self.edges if h != entity_id and t != entity_id]
    #     self.set_edges(edges)

    #     self.entities.remove(self.id_to_entity[entity_id])
    #     del self.id_to_entity[entity_id]

        # self.set_all_degrees() 
    
    '''
    Return subgraph containing edges of only given relation types
    '''
    def relation_subset_graph(self, relation_types):
        edges = [(h, r, t) for h, r, t in self.edges if r in relation_types]
        
        entities = set([h for (h, r, t) in edges] + [t for (h, r, t) in edges])
        relations = set([r for (h, r, t) in edges])

        entities = set([Entity(id=ent) for ent in entities])

        return KG(
            entities=entities,
            relation_types=relations,
            edges=edges
        )

    '''
    Return random sample of k edges of given relation type
    '''
    def sample_relations_of_type(self, relation_type, k=3):
        edges = [(h, r, t) for h, r, t in self.edges if r == relation_type]
        return random.sample(edges, k)

    def plot(self, out_path='kg.png'):
        import networkx as nx
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame({'head': self.head, 'relation': self.relation, 'tail': self.tail})
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(row['head'], row['tail'], label=row['relation'])

        pos = nx.spring_layout(G, seed=42, k=0.9)
        labels = nx.get_edge_attributes(G, 'label')
        plt.figure(figsize=(12, 10))
        nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
        plt.title('Knowledge Graph')
        plt.savefig(out_path)

    def get_out_neighbors(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]

        if entity.out_degree == -1:
            entity.out_neighbors = [(self.relation[i], self.tail[i]) for i in range(len(self.head)) if self.head[i] == entity.id]
            entity.out_degree = len(entity.out_neighbors)

        return entity.out_neighbors

    def set_out_degree(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]

        entity.out_neighbors = [(self.relation[i], self.tail[i]) for i in range(len(self.head)) if self.head[i] == entity.id]
        entity.out_degree = len(entity.out_neighbors)

    def get_out_degree(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
        return entity.out_degree
    
    def set_in_degree(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]

        entity.in_neighbors = [(self.head[i], self.relation[i]) for i in range(len(self.head)) if self.tail[i] == entity.id]
        entity.in_degree = len(entity.in_neighbors)

    def get_in_degree(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
        return entity.in_degree
    
    def merge_node(self, node1, node2):
        for i in range(len(self.edges)):
            if self.edges[i][0] == node2:
                self.edges[i] = (node1, self.edges[i][1], self.edges[i][2])
            if self.edges[i][2] == node2:
                self.edges[i] = (self.edges[i][0], self.edges[i][1], node1)
        
        self.set_edges(self.edges)

    def set_all_degrees(self):
        for entity in self.entities:
            entity.in_degree = 0
            entity.out_degree = 0
            entity.in_neighbors = []
            entity.out_neighbors = []

        for h, r, t in self.edges:
            head = self.id_to_entity[h]
            tail = self.id_to_entity[t]

            head.out_degree += 1
            head.out_neighbors.append((r, t))

            tail.in_degree += 1
            tail.in_neighbors.append((h, r))

    def build_entity_rel_value_matrix(self, entities_list, relations_list):
        entity2idx = {entity.id: i for i, entity in enumerate(entities_list)}
        relation2idx = {relation: i for i, relation in enumerate(relations_list)}

        print(len(entities_list))
        print(len(entity2idx))
        matrix = []
        for i in range(len(entities_list)):
            matrix.append([])
            for j in range(len(relations_list)):
                matrix[i].append([])
        print(len(matrix), len(matrix[0]))
        for i, (h, r, t) in enumerate(self.edges):
            # print(entity2idx[h], relation2idx[r])
            matrix[entity2idx[h]][relation2idx[r]].append(t)

        return matrix
            
            

    def get_two_hop(self):
        from tqdm import tqdm
        entities_list = list(self.entities)
        entity2idx = {entity.id: i for i, entity in enumerate(entities_list)}

        relations_list = list(self.relation_types)
        
        matrix = self.build_entity_rel_value_matrix(entities_list, relations_list)
        two_hop = []

        for i, ent1 in tqdm(enumerate(entities_list)):
            for j, rel in enumerate(relations_list):
                if len(matrix[i][j]) == 1:
                    ent2_id = matrix[i][j][0]
                    ent2_idx = entity2idx[ent2_id]
                    for k, rel2 in enumerate(relations_list):
                        if len(matrix[ent2_idx][k]) == 1:
                            ent3_id = matrix[ent2_idx][k][0]
                            two_hop.append((ent1.id, rel.id(), ent2_id, rel2.id(), ent3_id))

        return two_hop
    
    def get_sentence(self, edge):
        h,r,t = edge
        head = self.id_to_entity[h]
        relation = Relation.id_to_enum(r)
        tail = self.id_to_entity[t]
        return f'{relation.phrase(head.name)} {tail.name}.'

    def get_labeled_query(self, edge):
        h,r,t = edge
        head = self.id_to_entity[h]
        relation = Relation.id_to_enum(r)
        tail = self.id_to_entity[t]
        return {'query': f'{relation.phrase(head.name)} ', 'answer': tail.name}
    
    def set_relation_templates(self, method):
        self.relation_template_method = method
        if self.relation_template_method == 'fixed':
            self.get_fixed_templates()
        elif self.relation_template_method == 'random+subj_prec_obj':
            self.get_subj_prec_obj_templates()
        elif self.relation_template_method == 'random+all':
            self.get_all_templates()
        else:
            raise ValueError('Invalid relation template method')
        


    def get_subj_prec_obj_templates(self):
        subj_prec_obj_templates = {}
        for key, value in relations_templates.items():
            templates = []
            for template in value:
                if 'subject' in template and 'object' in template and template.index('subject') < template.index('object'):
                    templates.append(template)
            subj_prec_obj_templates[key] = templates

        self.templates = subj_prec_obj_templates

    def get_fixed_templates(self):
        self.templates = {key: value[0] for key, value in relations_templates.items()}

    def get_all_templates(self):
        self.templates = relations_templates
    

    def get_relation_template(self, head, rel, tail):
        method = self.relation_template_method
        if method == 'fixed':
            t = self.templates.get(rel, None)
            return t.replace('<subject>', head.name).replace('<object>', tail.name) + "."
        elif method == 'random+subj_prec_obj' or method == 'random+all':
            t = random.choice(self.templates.get(rel))
            return t.replace('<subject>', head.name).replace('<object>', tail.name) + "."
        else:
            raise ValueError('Invalid relation template method')
        # TODO: update paragraph functions with this

    def get_entity_paragraph(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
            
        sentences = []

        for rel, tail in entity.out_neighbors:
            relation = relation_id_to_name[rel]
            sentences.append(self.get_relation_template(entity, relation, self.id_to_entity[tail]))

        # append examples for which entity is tail
        for head, rel in entity.in_neighbors:
            relation = relation_id_to_name[rel]
            sentences.append(self.get_relation_template(self.id_to_entity[head], relation, entity))

        # shuffle sentences
        random.shuffle(sentences)
        return ' '.join(sentences)
    
    def get_entity_out_paragraph(self, entity):
        if isinstance(entity, str):
            entity = self.id_to_entity[entity]
            
        sentences = []

        for rel, tail in entity.out_neighbors:
            relation = relation_id_to_name[rel]
            sentences.append(self.get_relation_template(entity, relation, self.id_to_entity[tail]))
            
        # shuffle sentences
        random.shuffle(sentences)
        return ' '.join(sentences)
 
 


    # def get_entity_paragraph(self, entity):
    #     if isinstance(entity, str):
    #         entity = self.id_to_entity[entity]
            
    #     sentences = []

    #     for rel, tail in entity.out_neighbors:
    #         relation = Relation.id_to_enum(rel)
    #         sentences.append(f'{relation.phrase(entity.name)} {self.id_to_entity[tail].name}.')

    #     # append examples for which entity is tail
    #     for head, rel in entity.in_neighbors:
    #         relation = Relation.id_to_enum(rel)
    #         sentences.append(f'{relation.phrase(self.id_to_entity[head].name)} {entity.name}.')

    #     # shuffle sentences
    #     random.shuffle(sentences)
    #     return ' '.join(sentences)

    # def get_entity_out_paragraph(self, entity):
    #     if isinstance(entity, str):
    #         entity = self.id_to_entity[entity]
            
    #     sentences = []

    #     for rel, tail in entity.out_neighbors:
    #         relation = Relation.id_to_enum(rel)
    #         sentences.append(f'{relation.phrase(entity.name)} {self.id_to_entity[tail].name}.')

    #     # shuffle sentences
    #     random.shuffle(sentences)
    #     return ' '.join(sentences)
