import os
import numpy as np
# import pytorch
import torch
import torch.nn as nn

class Embedding(object):

    def __init__(self, embedd_dim, pad_id):
        # save parameters
        self.pad_id, self.embedd_dim = pad_id, embedd_dim
        # word2id map and embedding
        self.word2id = None
        self.embedding:nn.Embedding = None

    def load_csv(self, fpath):
        # clear word2id map
        self.word2id = {}
        # save first position for padding tensor
        # the padding tensor will not be trained or anything since its just a fill vector
        tensors = [torch.zeros(self.embedd_dim)]
        # load embedding tensors from file
        with open(csv_file, 'r', encoding='latin-1') as f:
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


    def load(self, words_file, embedd_file):
        # clear word2id map
        self.word2id = {}
        # load word-to-id map
        with open(words_file, "r") as f:
            # start enumerating at 1 to skip padding embedding at position 0
            for i, word in enumerate(f, 1):
                self.word2id[word] = i
        # load embedding and add padding embedding
        weight = torch.load(embedd_file, map_location='cpu')
        assert weight.size(1) == self.embedd_dim
        weight = torch.cat((torch.zeros((1, self.embedd_dim)), weight), dim=0)
        # create embedding
        self.embedding = nn.Embedding(
            num_embeddings=weight.size(0),
            embedding_dim=self.embedd_dim,
            _weight=weight
        )

    def embedd(self, ids):
        # make sure embedding is loaded before execution
        assert self.embedding is not None
        # set all padding ids
        ids[ids == self.pad_id] = 0
        return self.embedding(ids)


def train_embedding(fpath, dump_path, model="SimplE"):
    # imports
    from graph import SenticNetGraph
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    # load graph
    g = SenticNetGraph(fpath)
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
    # triples = triples[:100]
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
    print("Training Embedding...")
    results = pipeline(
        # data
        training_triples_factory=train_factory,
        testing_triples_factory=test_factory,
        # model
        model=model
    )
    print("Saving Embeddings...")
    # save entity and relation embeddings
    torch.save(results.model.entity_embeddings.weight, os.path.join(dump_path, "entities.bin"))
    torch.save(results.model.relation_embeddings.weight, os.path.join(dump_path, "relations.bin"))
    # save words in order matching the embeddings
    with open(os.path.join(dump_path, "entities.txt"), "w+", encoding='utf-8') as f:
        f.write('\n'.join([c.text for c in g.concepts]))
    # done with all
    print("Saved Embeddings to %s!" % dump_path)

if __name__ == '__main__':
    # train a knowledge graph embedding for senticnet graph
    train_embedding(
        "../data/senticnet/german/senticnet_de.rdf.xml", 
        "../data/senticnet/german/"
    )
    print("\nTrying to load new embedding...")
    e = Embedding(embedd_dim=200, pad_id=0)
    e.load(
        "../data/senticnet/german/entities.txt",
        "../data/senticnet/german/entities.bin"
    )
    print("Everything worked as intended!")
