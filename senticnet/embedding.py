# import pytorch
import torch
import torch.nn as nn

class Embedding(object):

    def __init__(self, csv_file, embedd_dim, pad_id):

        self.pad_id = pad_id
        self.word2id = {}
        # save first position for padding tensor
        # the padding tensor will not be trained or anything since its just a fill vector
        tensors = [torch.zeros(embedd_dim)]

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
                assert tensors[-1].size(0) == embedd_dim

        # create embedding from loaded tensors
        self.embedding = nn.Embedding(
            num_embeddings=len(tensors),    # add padding tensor
            embedding_dim=tensors[0].size(0),
            _weight=torch.stack(tensors, dim=0)
        )

    def embedd(self, ids):
        # set all padding ids
        ids[ids == self.pad_id] = 0
        return self.embedding(ids)