import unittest
import pyexml as pyml
import numpy as np
import torch
from test_helper import torch_equal

class TestDataset(unittest.TestCase):

    def test_dataset_tensor_creation1(self):

        # Create a dataset
        dataset = pyml.datasets.Dataset([1,2,3,4,5], [1,2,3,4,5])

        self.assertTrue(torch_equal(dataset.data, torch.tensor([1,2,3,4,5], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset.target, torch.tensor([1,2,3,4,5], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset.map_assignment.assignment, torch.tensor([0,1,2,3,4], device=pyml.backend.device, dtype=torch.long)))

        self.assertTrue(torch_equal(dataset[0], torch.tensor([[1],[1]], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset[1], torch.tensor([[2],[2]], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset[2], torch.tensor([[3],[3]], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset[3], torch.tensor([[4],[4]], device=pyml.backend.device, dtype=torch.long)))
        self.assertTrue(torch_equal(dataset[4], torch.tensor([[5],[5]], device=pyml.backend.device, dtype=torch.long)))


class TestMap(unittest.TestCase):

    def test_map_exceptions(self):

        #Should raise exception
        with self.assertRaises(Exception):
            pyml.datasets.Map([1,2,3,4,5,6,7,8,9,10])

        with self.assertRaises(Exception):
            pyml.datasets.Map([0,2,3,4,5,6,7,8,9,-1])

        #Create from set, should raise exception
        with self.assertRaises(Exception):
            pyml.datasets.Map(set([1,2,3,4,5,6,7,8,9,10]))


    def test_map_datatypes(self):

        # Create a map from list
        map = pyml.datasets.Map([0,1,2,3,4,5,6,7,8,9])
        self.assertTrue(torch_equal(map.assignment, torch.tensor([0,1,2,3,4,5,6,7,8,9], device=pyml.backend.device, dtype=torch.long)))

        # Create a map from tensor
        map = pyml.datasets.Map(torch.tensor([1,2,3,4,5,6,7,8,9,10]))
        self.assertTrue(torch_equal(map.assignment, torch.tensor([0,1,2,3,4,5,6,7,8,9], device=pyml.backend.device, dtype=torch.long)))

        #Create from numpy array
        map = pyml.datasets.Map(np.array([1,2,3,4,5,6,7,8,9,10]))
        self.assertTrue(torch_equal(map.assignment, torch.tensor([0,1,2,3,4,5,6,7,8,9], device=pyml.backend.device, dtype=torch.long)))

        #Create from integer
        map = pyml.datasets.Map(10)
        self.assertTrue(torch_equal(map.assignment, torch.tensor([0,1,2,3,4,5,6,7,8,9], device=pyml.backend.device, dtype=torch.long)))

        #Create from map
        map = pyml.datasets.Map(map)
        self.assertTrue(torch_equal(map.assignment, torch.tensor([0,1,2,3,4,5,6,7,8,9], device=pyml.backend.device, dtype=torch.long)))

    def test_map_query(self):

        # Create a map
        map = pyml.datasets.Map([0,1,2,3,4,5,6,7,8,9])
        self.assertTrue(torch_equal(map[0], torch.tensor(0, device=pyml.backend.device, dtype=torch.long)))

        #Reassign
        map.reassign([1,3,5,7,9,2,4,6,8,0])

        self.assertEqual(map.assignment, torch.tensor([1,3,5,7,9,2,4,6,8,0], device=pyml.backend.device, dtype=torch.long))
        self.assertEqual(map[4], torch.tensor(9, device=pyml.backend.device, dtype=torch.long))

# Run the tests
if __name__ == '__main__':
    unittest.main()