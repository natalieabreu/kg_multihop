import os, sys
import json
import random
import pickle
import numpy as np
import multiprocessing

from tqdm import tqdm
from functools import partial
from scipy.optimize import linprog


from kg.kg2 import KG
from kg.expert import Expert

class DatasetGenerator():
    def __init__(self, ground_truth, clusters):
        self.ground_truth_ = ground_truth
        self.clusters_ = clusters
        self.num_experts_ = None
        self.experts_ = []

        self.node2cluster_ = {}
        if clusters is not None:
            for idx, cluster in enumerate(clusters):
                for node in cluster:
                    self.node2cluster_[node] = idx

        self.dataset_statistics = {}

    def save_generator_to_pkl(self, path, expert_args):
        obj = {
            'generator': self,
            'expert_args': expert_args
        }
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def generate_dataset(self, num_samples, entity_sampling_strategy='uniform', paragraph='entity'):
        if len(self.experts_) == 0:
            raise ValueError('No experts generated. Please generate experts first.')
        
        samples = []
        dataset_statistics = {'entity_counts': {}}
        for expert in tqdm(self.experts_, position=0, desc='Generating samples'):
            generated = 0
            with tqdm(total=num_samples // self.num_experts_, position=1, leave=False) as pbar:
                while generated < num_samples // self.num_experts_:
                    node = expert.sample_node(entity_sampling_strategy)
                    if paragraph == 'entity':
                        sample = expert.kg_.get_entity_paragraph(node)
                    elif paragraph == 'entity_out':
                        sample = expert.kg_.get_entity_out_paragraph(node)
                    else:
                        raise ValueError('Invalid paragraph strategy: {}'.format(paragraph))
                    if sample == '':
                        continue
                    samples.append({'text': sample})

                    if node not in dataset_statistics['entity_counts']:
                        dataset_statistics['entity_counts'][node] = 0
                    dataset_statistics['entity_counts'][node] += 1
                    pbar.update(1)
                    generated+=1

        return {'samples': samples, 'statistics': dataset_statistics}
    
    def save_dataset(self, dataset, path):
        with open(path, 'r') as f:
            obj = json.load(f)
        obj.update(dataset)

        with open(path, 'w') as f:
            json.dump(obj, f)

    def generate_and_save_dataset(self, num_samples, entity_sampling_strategy='uniform', paragraph='entity', path='dataset.json'):
        dataset = self.generate_dataset(num_samples, entity_sampling_strategy, paragraph)
        self.save_dataset(dataset, path)



    def generate_experts(self, strategy, num_experts, **kwargs):
        self.num_experts_ = num_experts
        if strategy == 'denoising':
            self.denoising(**kwargs)
        elif strategy == 'skill_selection':
            self.skill_selection(**kwargs)
        elif strategy == 'random_walk':
            self.random_walk(**kwargs)
        else:
            raise ValueError('Invalid strategy: {}'.format(strategy))
        
    def generate_expert_denoising_(self, _, entities_per_expert=500, confidence=0.6):
            selected_cluster_indices = self.select_clusters(entities_per_expert)
            confidence_scores = np.ones(len(selected_cluster_indices)) * confidence
            expertise_vector = np.zeros(len(self.clusters_))
            for idx, cluster_idx in enumerate(selected_cluster_indices):
                expertise_vector[cluster_idx] = confidence_scores[idx]

            # modify graph
            kg = self.get_modified_graph(self.ground_truth_, expertise_vector)
        
            expert = Expert(expertise_vector, kg)
            sys.stdout.flush()

            return expert
        
    def denoising(self, entities_per_expert=500, confidence=0.6):
        print('Generating experts with denoising strategy...')
        print('Entities per expert:', entities_per_expert, 'Confidence:', confidence)   

        with multiprocessing.Pool(24) as p:
            self.experts_ = list(tqdm(p.imap(
                    partial(self.generate_expert_denoising_, entities_per_expert=entities_per_expert, confidence=confidence), 
                    range(self.num_experts_)
                ), total=self.num_experts_))
        
        # self.experts_ = []
        # for item in r:
        #     self.experts_.append(item)
        print('Generated experts:', len(self.experts_))
            


    def skill_selection(self, entities_per_expert=500, knowledge_per_expert=400):
        self.experts_ = []
        for _ in tqdm(range(self.num_experts_)):
            selected_cluster_indices = self.select_clusters(entities_per_expert)
            confidence_scores = self.get_confidence_scores_lp(selected_cluster_indices, knowledge_per_expert)
            expertise_vector = np.zeros(len(self.clusters_))
            for idx, cluster_idx in enumerate(selected_cluster_indices):
                expertise_vector[cluster_idx] = confidence_scores[idx]

            # modify graph
            kg = self.get_modified_graph(self.ground_truth_, expertise_vector)
            expert = Expert(expertise_vector, kg)
            self.experts_.append(expert)

    def random_walk(self, facts_per_expert=10000):
        print('Generating experts with random walk strategy...')
        print('Facts per expert:', facts_per_expert)   

        with multiprocessing.Pool(24) as p:
            self.experts_ = list(tqdm(p.imap(
                    partial(self.generate_expert_random_walk_, facts_per_expert=facts_per_expert), 
                    range(self.num_experts_)
                ), total=self.num_experts_))

        print(f'Generated {len(self.experts_)} experts.',)
        

    def generate_expert_random_walk_(self, _, facts_per_expert=10000):
        start = np.random.choice(list(self.ground_truth_.get_all_entities()))
        edges = self.ground_truth_.random_walk_edge_set(start, facts_per_expert)
        other_edges = [(h, r, t) for h, r, t in self.ground_truth_.edges if (h, r, t) not in edges]
        assert len(other_edges) + len(edges) == len(self.ground_truth_.edges)

        # scramble all the other edges
        wrong_edges = self.scramble_edges(other_edges)
        edges += wrong_edges

        kg = KG(self.ground_truth_.entities, self.ground_truth_.relation_types, edges=edges)
        
        expert = Expert(None, kg)
        return expert


    def select_clusters(self, N):
        """
        Select a set of clusters such that the total number of entities is around N.

        :param clusters: A list of lists of entities where each list represents a cluster.
        :param N: The target number of entities.
        :return: A list of selected cluster indices.
        """
        # Shuffle the clusters to ensure random selection
        cluster_indices = np.arange(len(self.clusters_))
        np.random.shuffle(cluster_indices)

        selected_indices = []
        current_total = 0

        for idx in cluster_indices:
            cluster_size = len(self.clusters_[idx])
            if current_total + cluster_size <= N:
                selected_indices.append(idx)
                current_total += cluster_size

            if current_total >= N:
                break

        # print('Selected clusters:', selected_indices)
        return selected_indices
    
    def get_confidence_scores_lp(self, selected_cluster_indices, K):
        '''
        Get the confidence scores for the selected clusters.
        :param selected_cluster_indices: A list of selected cluster indices.
        :param K: The target amount of knowledge.'''
        coefficients_equalities = [len(self.clusters_[i])for i in selected_cluster_indices]  # require \sum_{|C_i|*c_i} = K
        constants_equalities = [K]
        bounds = (0, 1)  # require 0 <= c_i <= 1

        coefficients_min_y = [-1]*len(selected_cluster_indices)  # maximize \sum_{c_i} 
        res = linprog(coefficients_min_y,
                    A_eq=coefficients_equalities,
                    b_eq=constants_equalities,
                    bounds=[bounds for _ in selected_cluster_indices])
        return res.x
    
    def get_modified_graph(self, kg, expertise_vector):
        '''
        Modify the graph according to the expertise vector.
        '''
        relation2tailentities = kg.get_tail_entities_by_relation()
        relation2headentities = kg.get_head_entities_by_relation()

        new_edges = []
        count = 0
        for h,r,t in kg.edges:
            head_cluster_idx = self.node2cluster_[h]
            tail_cluster_idx = self.node2cluster_[t]
            c_head = expertise_vector[head_cluster_idx]
            c_tail = expertise_vector[tail_cluster_idx]
            
            if random.random() < 1-max(c_head, c_tail):
                count += 1
                if c_head == c_tail: # randomly choose one to change
                    c_head += random.random()-0.5

                if c_head > c_tail: # swap out tail
                    tail_options = relation2tailentities[r]
                    tail_distr = [1-expertise_vector[self.node2cluster_[t]] for t in tail_options]
                    tail_distr = np.array(tail_distr)/sum(tail_distr)
                    # sample random tail acc to tail_distr
                    sampled_tail = np.random.choice(tail_options, p=tail_distr)
                    new_edges.append((h, r, sampled_tail))
                else: # swap out head
                    head_options = relation2headentities[r]
                    head_distr = [1-expertise_vector[self.node2cluster_[t]] for h in head_options]
                    head_distr = np.array(head_distr)/sum(head_distr)
                    # sample random head acc to head_distr
                    sampled_head = np.random.choice(head_options, p=head_distr)
                    new_edges.append((sampled_head, r, t))
            else:
                new_edges.append((h, r, t))

        print(f'{os.getpid()}: # of edges swapped: {count}')

        return KG(kg.entities, kg.relation_types, edges=new_edges, relation_template_method=kg.relation_template_method)
        

    def scramble_edges(self, edges):
        '''
        Scramble the edges.
        '''
        relation2tailentities = self.ground_truth_.get_tail_entities_by_relation()
        relation2headentities = self.ground_truth_.get_head_entities_by_relation()
        for idx, (h, r, t) in enumerate(edges):
            if random.random() < 0.5:
                tail_options = relation2tailentities[r]
                sampled_tail = np.random.choice(tail_options)
                edges[idx] = (h, r, sampled_tail)
            else:
                head_options = relation2headentities[r]
                sampled_head = np.random.choice(head_options)
                edges[idx] = (sampled_head, r, t)

        return edges
    


    