import pyexlab as pylab

from .neuralnetsubject import NeuralNetSubject

from .. import datasets
from .. import models
from .. import trainers
from .. import geometry

import torch
import os
import copy
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors


class MetricStaticFFWDSubject(NeuralNetSubject):

    def __init__(self, domain_path, image=None, metric_precompute_path=None, connectivity = 'random_partial', v_ratio = 0.2, train_ratio = 0.7, subspace_ratio = 0.2, device=torch.device("cpu")):
        
        self.domain_path = domain_path
        self.image = image
        self.metric_precompute_path = metric_precompute_path
        self.connectivity = connectivity
        self.v_ratio = v_ratio
        self.train_ratio = train_ratio
        self.subspace_ratio = subspace_ratio
        self.device = device

        self.load_npy_datasets()
        self.init_connectivity()
        self.init_datasets()
        self.init_models()
        self.init_trainers()
        self.build_subject()


    def build_subject(self, name = "Metric Static FFWD Map "):

        super().__init__(self.metric_coach, alt_name= name + self.connectivity)

        self.test_dict['Info']['Validation Nodes'] = self.node_subset[0]
        self.test_dict['Info']['Training Nodes'] = self.node_subset[1]
        self.test_dict['Info']['Training Edges'] = self.edge_subset[0]
        self.test_dict['Info']['Testing Edges'] = self.edge_subset[1]
        self.test_dict['Info']['Connectivity'] = self.connectivity

        if self.connectivity == 'k-nearest':
            self.test_dict['Info']['k'] = int(len(self.node_subset[1]) * self.subspace_ratio) 

    def k_nearest_subspace(self, X, k, train_ratio):

        #X : graph of sample distances
        train_graph = np.zeros(X.shape)
        test_graph = np.zeros(X.shape)
        k_del = int((k / train_ratio))

        #Returns sorted list of indeces along axis=1
        X_sorted_idx = np.array(np.argpartition(X, 1, axis=1))

        train_idx = X_sorted_idx[:, :k]
        train_graph[np.arange(train_graph.shape[0])[:,None], train_idx] = 1

        test_idx = X_sorted_idx[:, k:k_del]
        test_graph[np.arange(test_graph.shape[0])[:,None], test_idx] = 1

        mst = minimum_spanning_tree(X).toarray()

        train_graph += mst
        test_graph += mst

        train_idx_multi = np.array(np.nonzero(train_graph))
        test_idx_multi = np.array(np.nonzero(test_graph))

        train_idx_flat = np.ravel_multi_index(train_idx_multi, X.shape)
        test_idx_flat = np.ravel_multi_index(test_idx_multi, X.shape)


        return train_idx_flat, test_idx_flat

    def init_connectivity(self):

        if self.connectivity == 'full':
            self.edge_subset = [None, None]

        elif self.connectivity == 'random_partial' or self.precompute is None:

            total_datasize = int(self.num_edges * self.subspace_ratio)
            cutoff_idx = int(total_datasize * self.train_ratio)

            self.edge_subset = datasets.utils.split_indeces(self.num_edges, cutoff_idx, total_datasize)

        elif self.connectivity == 'k-nearest':
            self.k = int(len(self.node_subset[1]) * self.subspace_ratio) 
            pre_idx = tuple(np.meshgrid(self.node_subset[1], self.node_subset[1], indexing='ij'))
            self.edge_subset = self.k_nearest_subspace(np.squeeze(self.precompute[pre_idx]), self.k, self.train_ratio)
            
    def load_npy_datasets(self):
        #Load Datasets
        self.domain_set = np.load(self.domain_path)
        self.num_nodes = len(self.domain_set)
        self.image_set = self.image
        self.v_cutoff_idx = int(self.num_nodes * self.v_ratio)
        self.node_subset = datasets.utils.split_indeces(self.num_nodes, self.v_cutoff_idx, self.num_nodes)
        self.num_edges = len(self.node_subset[1])**2

        if self.image is None:
            self.image_set = np.zeros(self.domain_set.shape)
        elif type(self.image) is str:
            self.image_set = np.load(self.image)
        elif type(self.image) is int:
            self.image_set = np.zeros([self.domain_set.shape[0], self.image])

        if self.metric_precompute_path is not None:
            self.precompute = np.load(self.metric_precompute_path)
        else:
            self.precompute = None

    def init_datasets(self):
        
        
        #Start Main init
        self.metric_dataset = datasets.MetricSpaceDataset(metric = None, space = self.domain_set, precompute = self.precompute, subspace=self.node_subset[1], device=self.device)

        #Map datasets
        self.metric_training_dataset = torch.utils.data.Subset(self.metric_dataset, self.edge_subset[0])
        self.metric_testing_dataset = torch.utils.data.Subset(self.metric_dataset, self.edge_subset[1])

    def init_models(self):
        #Construct neural networks
        self.map_net = models.Simple_FFWDNet(self.domain_set.shape[1], self.image_set.shape[1], device=self.device)
        self.metric_net = models.MetricNet(self.map_net, geometry.metrics.torch_metrics.Euclidean(), device=self.device)

    def init_trainers(self):
        #Construct trainers
        self.metric_optimizer = torch.optim.SGD(self.metric_net.parameters(), lr=0.01)

        self.metric_trainer = trainers.Trainer(model = self.metric_net, dataset = self.metric_training_dataset, criterion = torch.nn.MSELoss(), optimizer = self.metric_optimizer, 
                                                scheduler = torch.optim.lr_scheduler.ExponentialLR(self.metric_optimizer, gamma=0.99),
                                                alt_name= "Metric Trainer")
        self.metric_tester  = trainers.Tester(self.metric_net, self.metric_testing_dataset, torch.nn.MSELoss(), alt_name= "Metric Tester")
        
        #Construct the coach that invokes the trainers at certain epochs
        self.trainer_schedule = trainers.Modulus_Schedule([-1, -1])
        self.trainer_list = [self.metric_trainer, self.metric_tester]

        self.metric_coach = trainers.Coach(trainers=self.trainer_list, trainer_schedule=self.trainer_schedule)

    def reset_connectivity(self, connectivity):
        self.connectivity = connectivity
        self.init_datasets()
        self.init_trainers()
        self.build_subject()
        

class MetricDynamicFFWDSubject(NeuralNetSubject):

    def __init__(self, domain_path, image=None, metric_precompute_path=None, connectivity = 'random_partial', v_ratio = 0.2, train_ratio = 0.7, subspace_ratio = 0.2):
        
        #Load Datasets
        domain_set = np.load(domain_path)
        image_set = image

        if image is None:
            image_set = np.zeros(domain_set.shape)
        elif type(image) is str:
            image_set = np.load(image)
        elif type(image) is int:
            image_set = np.zeros([domain_set.shape[0], image])

        if metric_precompute_path is not None:
            precompute = np.load(metric_precompute_path)
        else:
            precompute = None
        
        num_nodes = len(domain_set)
        
        v_cutoff_idx = int(num_nodes * v_ratio)
        node_subset = datasets.utils.split_indeces(num_nodes, v_cutoff_idx, num_nodes)

        metric_dataset = datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute = precompute, subspace=node_subset[1])

        
        if connectivity == 'full':
            subset_idx = [None, None]

        elif connectivity == 'random_partial' or precompute is None:

            num_edges = len(metric_dataset)
            total_datasize = int(num_edges * subspace_ratio)
            cutoff_idx = int(total_datasize * train_ratio)

            subset_idx = datasets.utils.split_indeces(num_edges, cutoff_idx, total_datasize)

        elif connectivity == 'k-nearest':
            k = int(len(node_subset[1]) * subspace_ratio) 
            pre_idx = tuple(np.meshgrid(node_subset[1], node_subset[1], indexing='ij'))
            subset_idx = self.k_nearest_subspace(np.squeeze(precompute[pre_idx]), k, train_ratio)
            

        #Map datasets
        map_training_dataset = datasets.MapDataset(domain_set, image_set, subspace=subset_idx[0])
        map_testing_dataset = datasets.MapDataset(domain_set, image_set, subspace=subset_idx[1])

        #Metric datasets
        metric_training_dataset = datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=precompute, subspace=subset_idx[0])
        metric_testing_dataset = datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=precompute, subspace=subset_idx[1])

        #Construct neural networks
        map_net = models.Simple_FFWDNet(domain_set.shape[1], image_set.shape[1])
        metric_net = models.MetricNet(map_net, geometry.metrics.torch_metrics.Euclidean())

        #Construct trainers
        metric_optimizer = torch.optim.Adam(metric_net.parameters(), lr=0.0001)
        map_optimizer = torch.optim.Adam(map_net.parameters(), lr=0.0001)

        map_trainer = trainers.DynamicLSATrainer(map_net, map_training_dataset, torch.nn.MSELoss(), 
                                                        optimizer = map_optimizer, 
                                                        scheduler = torch.optim.lr_scheduler.ExponentialLR(metric_optimizer, gamma=0.99),
                                                        alt_name="Map Trainer", epoch_mod=10)
        map_tester  = trainers.Tester(map_net, map_testing_dataset, torch.nn.MSELoss(), alt_name="Map Tester")

        metric_trainer = trainers.Trainer(metric_net, metric_training_dataset, torch.nn.MSELoss(), metric_optimizer, 
                                                scheduler = torch.optim.lr_scheduler.ExponentialLR(metric_optimizer, gamma=0.99),
                                                alt_name= "Metric Trainer")
        metric_tester  = trainers.Tester(metric_net, metric_testing_dataset, torch.nn.MSELoss(), alt_name= "Metric Tester")

        #Construct the coach that invokes the trainers at certain epochs
        trainer_schedule = trainers.Modulus_Schedule([-1, -1, -1, -1])
        trainer_list = [metric_trainer, map_trainer, metric_tester, map_tester]

        metric_coach = trainers.Coach(trainers=trainer_list, trainer_schedule=trainer_schedule)

        super().__init__(metric_coach, alt_name="Metric Static FFWD Map")

    def k_nearest_subspace(self, X, k, train_ratio):

        #X : graph of sample distances
        train_graph = np.zeros(X.shape)
        test_graph = np.zeros(X.shape)
        k_del = int((k / train_ratio))

        #Returns sorted list of indeces along axis=1
        X_sorted_idx = np.array(np.argpartition(X, 1, axis=1))

        train_idx = X_sorted_idx[:, :k]
        train_graph[np.arange(train_graph.shape[0])[:,None], train_idx] = 1

        test_idx = X_sorted_idx[:, k:k_del]
        test_graph[np.arange(test_graph.shape[0])[:,None], test_idx] = 1

        mst = minimum_spanning_tree(X).toarray()

        train_graph += mst
        test_graph += mst

        train_idx_multi = np.array(np.nonzero(train_graph))
        test_idx_multi = np.array(np.nonzero(test_graph))

        train_idx_flat = np.ravel_multi_index(train_idx_multi, X.shape)
        test_idx_flat = np.ravel_multi_index(test_idx_multi, X.shape)


        return train_idx_flat, test_idx_flat

