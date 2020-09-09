import torch
import torch.nn as nn
# import knowledge-base
from ...kb.knowledge import KnowledgeBase, KnowledgeBaseRegistry
# import senticnet stuff
from .graph import SenticNetGraph
from .embedding import SenticNetEmbedding
from .concept_parser import ConceptParser
# import utils
import os
import re
import shutil
from .utils import reconstruct_from_wordpieces
# import stopwords
from nltk.corpus import stopwords

@KnowledgeBaseRegistry.instance.register('sentic-net')
class SenticNet(KnowledgeBase):

    def __init__(self, data_path='./data/senticnet/english', lang='english'):
        super(SenticNet, self).__init__(embedd_dim=200)

        # load stopwords
        self.lang = lang
        self.sw = stopwords.words(lang)

        # only load what is required
        # if mode in ['all', 'prepare']:
            # load parser and sentic-net graph
            # self.concept_parser = ConceptParser()

        # create embedder
        self.embedder = SenticNetEmbedding(self.embedd_dim, self.pad_id)
        self.embedder.load(data_path)

        # always load graph
        self.graph_file = os.path.join(data_path, "senticnet.rdf.xml")
        self.graph = SenticNetGraph(self.graph_file)
        # cache
        self.nodes = None

    @property
    def config(self) -> dict:
        # get base configuration and add to it
        config = super(SenticNet, self).config
        config.update({'lang': self.lang})
        # return
        return config

    def save(self, save_directory:str) -> None:
        # save embeddings
        self.embedder.save(save_directory)
        # copy graph file
        _, fname = os.path.split(self.graph_file)
        dst = os.path.join(save_directory, fname)
        if self.graph_file != dst:
            shutil.copyfile(self.graph_file, dst)

    @staticmethod
    def load(directory:str, config:dict):
        return SenticNet(directory, lang=config['lang'])

    def find_mentions(self, wordpiece_tokens):
        # return self.find_concept_mentions(wordpiece_tokens)
        return self.find_token_mentions(wordpiece_tokens)

    def find_token_mentions(self, wordpiece_tokens):
        # reconstruct text and find tokens
        text, spans = reconstruct_from_wordpieces(wordpiece_tokens)
        concept_matches = [m for m in re.finditer('(\w)+', text.lower()) if m.group() not in self.sw]
        # check for concepts candidates
        if len(concept_matches) == 0:
            return {}
        # get concept-terms and spans
        concept_terms, concept_spans = zip(*[(m.group(), (m.start(), m.end())) for m in concept_matches]) 
        # get concept-nodes from senticnet-parser
        concept_nodes = [self.graph.get_concept(term) for term in concept_terms]
        # select all concepts contained in senticnet
        concept_mask = [node is not None for node in concept_nodes]
        concept_nodes = [node for node, valid in zip(concept_nodes, concept_mask) if valid]
        concept_terms = [term for term, valid in zip(concept_terms, concept_mask) if valid]
        concept_spans = [span for span, valid in zip(concept_spans, concept_mask) if valid]
        # no mentions found
        if len(concept_terms) == 0:
            return {}
        # bin all tokens in concepts
        concept_tokens = [[] for _ in range(sum(concept_mask))]
        cur_concept_idx = 0
        for i, (b, e) in enumerate(spans):
            # bin token in concepts
            cb, ce = concept_spans[cur_concept_idx]
            if (cb <= b) and (e <= ce):
                concept_tokens[cur_concept_idx].append(i)
            # next concept
            if e >= ce:
                cur_concept_idx += 1
            # all concepts processed
            if cur_concept_idx >= sum(concept_mask):
                break
        # build concept-token-ma
        return dict(zip(concept_terms, concept_tokens))

    def find_concept_mentions(self, wordpiece_tokens):
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
        # return
        return concept_token_map

    def find_candidates(self, mention):
        # get semantics from graph
        node = self.graph.get_concept(mention)
        semantics = self.graph.get_semantic_ids(node)
        # get text of semantics and return
        return [node.index] + semantics

    def get_prior(self, candidate_id):
        # use attention score of concept as prior probability
        return self.graph.get_concept_from_id(candidate_id).attention

    def id2entity(self, concept_id):
        return self.graph.get_concept_from_id(concept_id).text

    def embedd(self, node_ids):
        # get concepts from node-ids
        flat_node_ids = node_ids.flatten().tolist()
        concepts = [self.graph.get_concept_from_id(i) if i != self.pad_id else None for i in flat_node_ids]
        # get embeddings from concept-terms
        concept_terms = [c.text if c is not None else None for c in concepts]
        flat_embedding_ids = [self.embedder.word2id.get(w, 0) if w is not None else self.pad_id for w in concept_terms]
        embedding_ids = torch.tensor(flat_embedding_ids).long().view(node_ids.size())
        # return embeddings
        return self.embedder.embedd(embedding_ids)
