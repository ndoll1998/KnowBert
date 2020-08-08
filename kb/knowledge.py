import torch
from abc import ABC

class KnowledgeBase(ABC):
    """ Abstract base class for knowledge bases/graphs """

    def __init__(self, embedd_dim:int, pad_id:int =-1):
        """ Specify the embedding dimension and the fill value for padding """
        # save values
        self.embedd_dim = embedd_dim
        self.pad_id = pad_id

    def find_mentions(self, tokens):
        """ find all entity mention spans in given list of tokens. 
            This needs to return a dict mapping the mention to its token-span """
        raise NotImplementedError()

    def find_candidates(self, mention):
        """ get candidate entities from mention. 
            This needs to return a list of entity-ids that get passed to the embedding function.
            Also the find_candidates function gets called immidiatly after find_mentions for all found mentions
        """
        raise NotImplementedError()

    def embedd(self, entity_ids):
        """ embedd the given entities specified by their ids.
            This needs to return a pytorch tensor of size (*entity_ids, embedd_dim) 
        """
        raise NotImplementedError()

    def get_prior(self, entity_id):
        """ get prior probability of the given entity """
        raise NotImplementedError()

    def id2entity(self, entity_id):
        """ (Optional) get the entity term from it's id """
        raise NotImplementedError()
