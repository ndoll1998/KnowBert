import torch
import torch.nn as nn
# import knowledge-base
from kb.knowledge import KnowledgeBase
# import senticnet stuff
from .graph import SenticNetGraph
from .embedding import Embedding
from .concept_parser import ConceptParser
# import utils
from .utils import reconstruct_from_wordpieces


class SenticNet(KnowledgeBase):

    def __init__(self, mode="all"):
        super(SenticNet, self).__init__(embedd_dim=100)

        # only load what is required
        if mode in ['all', 'prepare']:
            # load parser and sentic-net graph
            self.concept_parser = ConceptParser()
            # cache
            self.nodes = None

        if mode in ['all', 'train']:
            # create embedder
            self.embedder = Embedding("data/senticnet/affectivespace.csv", self.embedd_dim, self.pad_id)

        # always load graph
        self.graph = SenticNetGraph("data/senticnet/senticnet5.rdf.xml")

    def find_mentions(self, wordpiece_tokens):
        # reconstruct text and find concepts
        text, spans = reconstruct_from_wordpieces(wordpiece_tokens)
        concepts = self.concept_parser.parse(text)
        # build concept terms - concepts can contain multiple tokens
        concept_terms = ['_'.join([t.text for t in c]) for c in concepts]
        # get concept-nodes from senticnet-parser
        concept_nodes = [self.graph.get_concept(term) for term in concept_terms]
        # select all concepts contained in senticnet
        concept_mask = [node is not None for node in concept_nodes]
        concept_nodes = [node for node, valid in zip(concept_nodes, concept_mask) if valid]
        concept_terms = [term for term, valid in zip(concept_terms, concept_mask) if valid]
        concepts = [c for c, valid in zip(concepts, concept_mask) if valid]
        # get indices of terms that build a concept
        concept_spans = [[(t.begin, t.end) for t in c] for c in concepts]
        concept_idx = [[int(t.index) for t in c] for c in concepts]
        # flatten
        concept_spans_flat = sum(concept_spans, [])
        concept_idx_flat = sum(concept_idx, [])
        # remove doubles
        concept_idx_clean = list(set(concept_idx_flat))
        concept_spans_clean = [concept_spans_flat[concept_idx_flat.index(i)] for i in concept_idx_clean]
        # get word-piece concept spans
        concept_wp_spans_clean = [
            [i for i, (wp_begin, wp_end) in enumerate(spans) if (wp_begin >= begin) and (wp_end <= end)] 
            for (begin, end) in concept_spans_clean
        ]
        # map concept to word-piece tokens
        concept_token_map = {
            term: sum([concept_wp_spans_clean[concept_idx_clean.index(i)] for i in idx], [])
            for term, idx in zip(concept_terms, concept_idx)
        }
        # cache nodes
        self.nodes = dict(zip(concept_terms, concept_nodes))
        # return
        return concept_token_map

    def find_candidates(self, mention):
        # check if node for current mention is caches
        node = self.nodes[mention] if mention in self.nodes else self.graph.get_concept(mention)
        # get semantics from node
        semantics = self.graph.get_semantic_ids(node)
        # get text of semantics and return
        return [node.index] + semantics

    def get_prior(self, embedd_id):
        return 1

    def id2entity(self, entity_id):
        return self.embedder.id2word[entity_id]

    def embedd(self, node_ids):
        # get concepts from node-ids
        flat_node_ids = node_ids.flatten().tolist()
        concepts = [self.graph.get_node_from_id(i) if i != self.pad_id else None for i in flat_node_ids]
        # get embeddings from concept-terms
        concept_terms = [c.text if c is not None else None for c in concepts]
        flat_embedding_ids = [self.embedder.word2id.get(w, 0) if w is not None else self.pad_id for w in concept_terms]
        embedding_ids = torch.tensor(flat_embedding_ids).long().view(node_ids.size())
        # return embeddings
        return self.embedder.embedd(embedding_ids)
