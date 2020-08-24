import torch
from abc import ABC
from functools import wraps

class KnowledgeBase(ABC):
    """ Abstract base class for knowledge bases/graphs """

    def __init__(self, embedd_dim:int, pad_id:int =-1):
        """ Specify the embedding dimension and the fill value for padding """
        # save values
        self.embedd_dim = embedd_dim
        self.pad_id = pad_id

    @property
    def config(self) -> dict:
        """ Get the configuration of the knowledge base. """
        return {
            'type': KnowledgeBaseManager.instance.get_name_from_type(self.__class__), 
            'embedd_dim': self.embedd_dim, 
            'pad_id': self.pad_id
        }

    def save(self, save_directory:str) -> None:
        """ Save all necessary data to directory """
        raise NotImplementedError()

    @staticmethod
    def load(self, directory:str, config:dict):
        """ Load knowledge base from a save directory and configuration dictionary """
        raise NotImplementedError()

    def find_mentions(self, tokens):
        """ find all entity mention spans in given list of tokens. 
            This needs to return a dict mapping a mention term to its token-ids """
        raise NotImplementedError()

    def find_candidates(self, mention):
        """ get candidate entities from mention. 
            This needs to return a list of entity-ids that get passed to the embedd function.
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



class __KnowledgeBaseManagerType(type):
    """ Singleton Meta Class """

    @property
    def instance(self):
        return KnowledgeBaseManager.get_instance()

    def __call__(self):
        return KnowledgeBaseManager.get_instance()


class KnowledgeBaseManager(metaclass=__KnowledgeBaseManagerType):
    """ Singleton Manager to manage all Knowledge Base types """

    __instance = None

    def __init__(self):
        # create empty dict to store all knowledge base types
        self.name2type = {}
        self.type2name = {}

    @staticmethod
    def get_instance():
        # only initialize once
        if KnowledgeBaseManager.__instance is None:
            KnowledgeBaseManager.__instance = object.__new__(KnowledgeBaseManager)
            KnowledgeBaseManager.__instance.__init__()
        # return instance
        return KnowledgeBaseManager.__instance

    def register(self, name:str):
        """ Decorator function to register knowledge base types to the manager """
        # check if name is already in use
        if name in self.name2type:
            raise RuntimeError("Name %s is already used by %s!" % (name, self.name2type[name]))

        # create decorator function
        def wrapper(cls):
            # add type to dict
            self.name2type[name] = cls 
            self.type2name[cls] = name
            return cls

        return wrapper

    def get_type_from_name(self, name:str):
        # check if name is registered
        if name not in self.name2type:
            raise RuntimeError("No knowledge base with name %s registered! Did you forget to import it?" % name)
        # return type
        return self.name2type[name]
        
    def get_name_from_type(self, type:type):
        # check if name is registered
        if type not in self.type2name:
            raise RuntimeError("No knowledge base of type %s registered!" % type)
        # return type
        return self.type2name[type]