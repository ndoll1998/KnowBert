import os
import numpy as np
# import pytorch
import torch
import torch.nn as nn
# utils
from collections import OrderedDict

class SenticNetEmbedding(object):

    def __init__(self, embedd_dim, pad_id=None):
        # save parameters
        self.pad_id, self.embedd_dim = pad_id, embedd_dim
        # word2id map and embedding
        self.word2id = None
        self.embedding:nn.Embedding = None

    def embedd(self, ids):
        # make sure embedding is loaded before execution
        assert self.embedding is not None
        # set all padding ids
        ids[ids == self.pad_id] = 0
        return self.embedding(ids)

    def load_csv(self, fpath):
        # clear word2id map
        self.word2id = OrderedDict()
        # save first position for padding tensor
        # the padding tensor will not be trained or anything since its just a fill vector
        tensors = [torch.zeros(self.embedd_dim)]
        # load embedding tensors from file
        with open(fpath, 'r', encoding='latin-1') as f:
            for row in f:
                # split and create tensor
                items = row.split(',')
                word, tensor = items[0], items[1:]
                tensor = torch.tensor([float(v) for v in tensor])
                # update
                self.word2id[word] = len(tensors)
                tensors.append(tensor)
                # check embedding dimension
                assert tensors[-1].size(0) == self.embedd_dim

        # create embedding from loaded tensors
        self.embedding = nn.Embedding(
            num_embeddings=len(tensors),    # add padding tensor
            embedding_dim=self.embedd_dim,
            _weight=torch.stack(tensors, dim=0)
        )

    def load(self, path):
        print("Loading SenticNet Embeddings... (%s)" % path)
        # clear word2id map
        self.word2id = OrderedDict()
        # load word-to-id map
        with open(os.path.join(path, "entities.txt"), "r") as f:
            # start enumerating at 1 to skip padding embedding at position 0
            for i, word in enumerate(f, 1):
                self.word2id[word] = i
        # load embedding and add padding embedding
        weight = torch.load(os.path.join(path, 'entities.bin'), map_location='cpu')
        assert weight.size(1) == self.embedd_dim
        weight = torch.cat((torch.zeros((1, self.embedd_dim)), weight), dim=0)
        # create embedding
        self.embedding = nn.Embedding(
            num_embeddings=weight.size(0),
            embedding_dim=self.embedd_dim,
            _weight=weight
        )

    def train_embedding(self, g, model="SimplE"):
        # pykeen
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory

        # create pseudo-nodes to enucode node attributes
        pleasent, not_pleasent = len(g.concepts), len(g.concepts) + 1
        sensitiv, not_sensitive = len(g.concepts) + 2, len(g.concepts) + 3
        # build triples
        triples = []
        for c in g.concepts:
            # actual connections
            triples.extend(([c.index, 'semantic', j] for j in g.get_semantic_ids(c)))
            # encode attributes by binning
            if c.pleasentness != 0:
                triples.append([c.index, 'pleasent', pleasent if c.pleasentness > 0 else not_pleasent])
            if c.sensitivity != 0:
                triples.append([c.index, 'sensitiv', sensitiv if c.sensitivity > 0 else not_sensitive])
        triples, n = np.asarray(triples), len(triples)
        print("Number of Triples (Train/Total): %i/%i" % (int(0.8 * n), n))
        # create mask for training and testing separation
        train_mask = np.full(n, False)
        train_mask[:int(n * 0.8)] = True
        np.random.shuffle(train_mask)
        # separate into training and testing
        train_triples = triples[train_mask]
        test_triples = triples[~train_mask]
        # create triples factories
        train_factory = TriplesFactory(triples=train_triples)
        test_factory = TriplesFactory(triples=test_triples)
        # create and run pipeline
        results = pipeline(
            # data
            training_triples_factory=train_factory,
            testing_triples_factory=test_factory,
            # model
            model=model,
            model_kwargs={
                "embedding_dim": self.embedd_dim,
                "automatic_memory_optimization": True
            }
        )
        # get embedding tensor - remove pseudo nodes
        weight = results.model.entity_embeddings.weight[:len(g.concepts), ...].cpu()
        
        # update word2id
        words = [c.text for c in g.concepts]
        self.word2id = OrderedDict( zip(words, range(1, len(words) + 1)) )  # 0th element is padding
        # update embeddings - add padding embedding at position 0
        self.embedding = nn.Embedding(
            num_embeddings=len(words) + 1,
            embedding_dim=self.embedd_dim,
            _weight=torch.cat((torch.zeros((1, self.embedd_dim)), weight), dim=0)
        )
        # return results
        return results

    def save(self, dump_path):
        # save entity and relation embeddings without padding embedding
        torch.save(self.embedding.weight[1:, ...], os.path.join(dump_path, "entities.bin"))
        # save words in order matching the embeddings
        with open(os.path.join(dump_path, "entities.txt"), "w+", encoding='utf-8') as f:
            f.write('\n'.join(self.word2id.keys()))

if __name__ == '__main__':
    # import graph
    from graph import SenticNetGraph

    # load graph
    graph = SenticNetGraph("../data/senticnet/german/senticnet_de.rdf.xml")
    # train a knowledge graph embedding for senticnet graph
    embedding = Embedding(embedd_dim=200)

    print("Training Embedding...")
    results = embedding.train_embedding(graph, model="SimplE")
    eval_results = results.metric_results.to_flat_dict()
    for metric, value in eval_results.items():
        print(metric.ljust(30, ' '), value)

    print("Saving Embedding...")
    embedding.save("../data/senticnet/german")

