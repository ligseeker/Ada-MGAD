import unittest

import torch
from torch_geometric.utils import dense_to_sparse, remove_self_loops

from src.model_util import adj2adj


def reference_adj2adj(graph, batch_size, window_size, zdim):
    graph1 = graph.squeeze(0).squeeze(0).repeat(batch_size, window_size, 1, 1).reshape(
        -1, graph.shape[-2], graph.shape[-1]
    )
    adj0, adj1, features = [], [], []
    node_adj = dense_to_sparse(graph1)[0]
    node_features = graph.unsqueeze(-1).repeat(1, 1, zdim)
    for node_index in range(node_adj.shape[1]):
        incoming = torch.argwhere(node_adj[1] == node_index)
        outgoing = torch.argwhere(node_adj[0] == node_index)
        adj0.append(incoming.repeat(1, outgoing.shape[0]).reshape(-1))
        adj1.append(outgoing.repeat(incoming.shape[0], 1).reshape(-1))
        features.append(torch.ones(outgoing.shape[0] * incoming.shape[0]) * node_index)
    edge_adj, edge_features = remove_self_loops(
        torch.stack([torch.concat(adj0), torch.concat(adj1)], dim=0),
        torch.concat(features),
    )
    return node_adj, node_features, edge_adj, edge_features


class AdjacencyConstructionTests(unittest.TestCase):
    def test_optimized_construction_is_elementwise_reference_equivalent(self):
        graph = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        expected = reference_adj2adj(graph, 2, 2, 4)
        actual = adj2adj(graph, 2, 2, 4)
        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))


if __name__ == "__main__":
    unittest.main()
