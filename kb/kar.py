""" Knowledge Attention and Recontextualization """

import torch
import torch.nn as nn
# import transformers 
from transformers import BertConfig
from transformers.modeling_bert import BertLayerNorm, BertEncoder, BertSelfOutput, BertOutput, BertIntermediate
# import utils
import math
from .utils import (
    match_shape_2d,
    pseudo_inverse, 
    init_bert_weights,
    bert_extended_attention_mask
)


""" Mention Span Representation """

class SelfAttentiveSpanPooler(nn.Module):

    def __init__(self, input_dim:int):
        # initialize Module
        super(SelfAttentiveSpanPooler, self).__init__()
        self.attention = nn.Linear(input_dim, 1, bias=False)
        # save dimension size
        self.dim = input_dim

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.attention, 0.02)

    def forward(self, h, spans):
        # self-attentive span pooling
        shape = h.size()
        attention_logits = self.attention(h.view(-1, self.dim))
        attention_logits = attention_logits.view(*shape[:-1], 1)
        # gather spans
        idx = torch.arange(spans.size(0)).repeat(spans.size(1), 1).T.unsqueeze(-1)
        sequence_spans = h[idx, spans, ...]
        attention_spans = attention_logits[idx, spans, ...]
        # apply mask and softmax to attention spans
        attention_spans[spans == -1] = -10000
        attention_weight_spans = torch.softmax(attention_spans, dim=-2)
        attention_weight_spans[spans == -1] = 0
        # compute weighted sum of sequence spans and attention weights
        pooled = (sequence_spans * attention_weight_spans).sum(-2)

        # return pooled tensors
        return pooled

class MentionSpanRepresenter(nn.Module):
    
    def __init__(self, kb, max_mentions:int):
        super(MentionSpanRepresenter, self).__init__()
        # save values
        self.max_mentions = max_mentions
        self.embedd_dim = kb.embedd_dim
        # create all sub-modules
        self.pooler = SelfAttentiveSpanPooler(self.embedd_dim)
        self.span_repr_ln = BertLayerNorm(self.embedd_dim, eps=1e-5)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.span_repr_ln, 0.02)

    def forward(self, h_proj, mention_spans):
        # build the mention-span representations
        mention_span_reprs = self.pooler(h_proj, mention_spans)
        # apply layer normalization and return
        mention_span_reprs = self.span_repr_ln(mention_span_reprs)
        return mention_span_reprs
        

""" Entity Linking and Knowledge Enhanced Representation """

class EntityLinker(nn.Module):

    def __init__(self,  kb, span_encoder_config:BertConfig, hidden_dim:int =100):
        super(EntityLinker, self).__init__()
        # create all sub-components
        self.encoder = self._create_span_encoder(kb, span_encoder_config)
        self.score_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        # layer norms
        self.candidate_embs_ln = BertLayerNorm(kb.embedd_dim, eps=1e-5)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.score_mlp[0], 0.02)
        init_bert_weights(self.score_mlp[2], 0.02)
        init_bert_weights(self.candidate_embs_ln, 0.02)

    def _create_span_encoder(self, kb, span_encoder_config):
        # check if encoder should be used
        if span_encoder_config is None:
            # return identity function as encoder
            return lambda t, m, h: t
        # update values to match dimensions
        span_encoder_config.hidden_size = kb.embedd_dim
        # create config and encoder
        return BertEncoder(span_encoder_config)

    def forward(self, candidate_embs, candidate_mask, candidate_priors, mention_span_reprs, mention_span_mask):
        # apply layer norm to candidate embeddings
        candidate_embs = self.candidate_embs_ln(candidate_embs)
        # apply span-encoder
        head_mask = None if not isinstance(self.encoder, BertEncoder) else [None] * self.encoder.config.num_hidden_layers
        extended_mention_span_mask = bert_extended_attention_mask(mention_span_mask)

        mention_span_reprs = self.encoder(
            hidden_states=mention_span_reprs, 
            attention_mask=extended_mention_span_mask, 
            head_mask=head_mask
        )[0]

        # compute entity linking scores
        scores = (candidate_embs * mention_span_reprs.unsqueeze(-2)).sum(-1) / math.sqrt(candidate_embs.size(-1))
        scores_with_prior = torch.cat((scores.unsqueeze(-1), candidate_priors.unsqueeze(-1)), dim=-1)
        linking_scores = self.score_mlp(scores_with_prior).squeeze(-1)
        linking_scores = linking_scores.masked_fill(~candidate_mask.bool(), -10000.0)
        
        # return linking scores
        return linking_scores


class KnowledgeEnhancedRepresentation(nn.Module):

    def __init__(self, kb, threshold:float =None):
        super(KnowledgeEnhancedRepresentation, self).__init__()
        # save values
        self.threshold = threshold
        self.null_emb = None if threshold is None else nn.Parameter(torch.zeros(kb.embedd_dim))
        # dropout
        self.enhanced_embs_ln = BertLayerNorm(kb.embedd_dim, eps=1e-5)
        self.dropout = nn.Dropout(0.1)
        
        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.enhanced_embs_ln, 0.02)
        
    def forward(self, linking_scores, candidate_embs, mention_span_reprs, mention_mask):
        # compute weights from linking scores
        if self.threshold is not None:
            # apply threshold
            below_threshold = linking_scores < self.threshold
            linking_scores.masked_fill(below_threshold, -10000.0)
        # apply softmax
        normalized_linking_scores = torch.softmax(linking_scores, dim=-1)
        normalized_linking_scores = mention_mask.unsqueeze(-1).float().detach() * normalized_linking_scores

        # compute weighted sum of candidate embeddings
        entity_embs = (normalized_linking_scores.unsqueeze(-1) * candidate_embs).sum(-2)
        entity_embs = self.dropout(entity_embs)

        # handle no value above threshold
        if (self.threshold is not None) and (self.null_emb is not None):
            all_below_threshold = below_threshold.sum(-1) == linking_scores.shape[-1]
            entity_embs[all_below_threshold] = self.null_emb.unsqueeze(0)

        # compute enhanced span representations
        enhanced_span_reprs = mention_span_reprs + entity_embs
        enhanced_span_reprs = self.enhanced_embs_ln(enhanced_span_reprs)

        # compute entropy loss from linking scores
        probs = normalized_linking_scores[mention_mask]
        log_probs = torch.log(probs + 1e-5)
        entropy = -(probs * log_probs).sum(-1)

        # return enhanced representations and entropy of linking attention
        return enhanced_span_reprs, entropy


""" Recontextualization """

class SpanWordAttention(nn.Module):

    def __init__(self, config):
        super(SpanWordAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, entity_embeddings, entity_mask):
        # apply linear transformation
        mixed_key_layer = self.key(entity_embeddings)
        mixed_query_layer = self.query(hidden_states)
        mixed_value_layer = self.value(entity_embeddings)
        # transpose for further computations
        key_layer = self.transpose_for_scores(mixed_key_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # compute raw attention scores
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # apply attention mask
        attention_mask = bert_extended_attention_mask(entity_mask)
        attention_scores = attention_scores + attention_mask
        # apply softmax and dropout
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # contextualize values
        context_layer = attention_probs @ value_layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
        # return contextualized values and attention probs
        return context_layer, attention_probs


class SpanAttention(nn.Module):

    def __init__(self, config):
        super(SpanAttention, self).__init__()
        # create modules
        self.attention = SpanWordAttention(config)
        self.output = BertSelfOutput(config)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.attention, 0.02, (SpanWordAttention, ))
        init_bert_weights(self.output, 0.02)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        span_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        attention_output = self.output(span_output, hidden_states)
        return attention_output, attention_probs


class SpanAttentionLayer(nn.Module):

    def __init__(self, config):
        super(SpanAttentionLayer, self).__init__()
        # create modules
        self.attention = SpanAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.intermediate, 0.02)
        init_bert_weights(self.output, 0.02)

    def forward(self, hidden_states, entity_embeddings, entity_mask):
        attention_output, attention_probs = self.attention(hidden_states, entity_embeddings, entity_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class Recontextualizer(nn.Module):

    def __init__(self, kb, span_attention_config:BertConfig):
        super(Recontextualizer, self).__init__()
        # create modules
        self.span_attention_layer = self._create_span_attention_layer(kb, span_attention_config)

    def _create_span_attention_layer(self, kb, span_attention_config):
        # update values to match dimensions
        span_attention_config.hidden_size = kb.embedd_dim
        # create and return span-attention-layer
        return SpanAttentionLayer(span_attention_config)

    def forward(self, enhanced_span_reprs, h_proj, mention_mask):
        # apply span attention and return
        span_attention_output, _ = self.span_attention_layer(h_proj, enhanced_span_reprs, mention_mask)
        return span_attention_output


""" Knowledge Attention and Recontextualization component """

class KAR(nn.Module):

    def __init__(self, kb, 
        bert_config:BertConfig, 
        span_encoder_config:BertConfig, 
        span_attention_config:BertConfig,
        max_mentions:int, 
        max_mention_span:int,
        max_candidates:int,
        threshold:float =None, 
    ):
        # initialize super
        super(KAR, self).__init__()
        # save values
        self.max_mentions = max_mentions
        self.max_mention_span = max_mention_span
        self.max_candidates = max_candidates
        # save knowledge base and create caches-list
        self.kb = kb
        self.cache = None
        self.clear_cache()
        # projections from bert to kb and reversed
        self.bert2kb = nn.Linear(bert_config.hidden_size, kb.embedd_dim)
        self.kb2bert = nn.Linear(kb.embedd_dim, bert_config.hidden_size)
        # create all modules
        self.mention_span_representer = MentionSpanRepresenter(self.kb, self.max_mentions)
        self.entity_linker = EntityLinker(self.kb, span_encoder_config)
        self.enhanced_representation = KnowledgeEnhancedRepresentation(self.kb, threshold)
        self.recontextualizer = Recontextualizer(self.kb, span_attention_config)
        # layernorms and dropout
        self.dropout = nn.Dropout(0.1)
        self.output_ln = BertLayerNorm(bert_config.hidden_size, eps=1e-5)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # initialize module weights
        init_bert_weights(self.output_ln, 0.02)
        init_bert_weights(self.bert2kb, 0.02)

        # initialize weight of projection to be the (pseudo-) inverse of the reversed projection
        with torch.no_grad():
            # get weight of first projection and compute it's pseudo-inverse
            w = self.bert2kb.weight.data
            w_inv = pseudo_inverse(w)
            # apply w-inv to bias
            b = self.bert2kb.bias.data
            b_inv = w_inv @ b
            # update parameters
            self.kb2bert.weight.data.copy_(w_inv)
            self.kb2bert.bias.data.copy_(b_inv)

    def get_cache_and_mention_candidates(self, tokens):
        # get mentions and mention spans
        mentions = self.kb.find_mentions(tokens)
        mention_terms, mention_spans = zip(*mentions.items()) if len(mentions) > 0 else ([], [])
        mention_terms, mention_spans = mention_terms[:self.max_mentions], mention_spans[:self.max_mentions]
        n_mentions = len(mention_terms)

        # max-size for dimension 1
        shape = (self.max_mentions, max(self.max_mention_span, self.max_candidates))
        # build mention-spans tensor
        mention_spans = match_shape_2d(mention_spans, shape, -1).long()
        # get candidate entity-ids for each found mention
        all_candidate_ids = [self.kb.find_candidates(m) for m in mention_terms]
        all_candidate_mask = [[1] * len(ids) for ids in all_candidate_ids]
        all_candidate_priors = [[self.kb.get_prior(i) for i in ids] for ids in all_candidate_ids]
        # match shape
        all_candidate_ids = match_shape_2d(all_candidate_ids, shape, self.kb.pad_id)
        all_candidate_mask = match_shape_2d(all_candidate_mask, shape, 0)
        all_candidate_priors = match_shape_2d(all_candidate_priors, shape, 0)

        # build mention-candidate map
        mention_candidate_map = list(zip(mention_terms, all_candidate_ids))
        # stack all tensors to build cache
        tensors = (mention_spans.float(), all_candidate_ids.float(), all_candidate_mask.float(), all_candidate_priors.float())
        cache = torch.stack(tensors, dim=0).unsqueeze(0)
        # return cache and mention-candidate-map
        return cache, mention_candidate_map

    def clear_cache(self):
        # empty cache
        self.cache = torch.empty((0, 4, self.max_mentions, max(self.max_mention_span, self.max_candidates))).float()

    def stack_caches(self, *caches):
        # stack all given caches on current cache
        self.cache = torch.cat((self.cache.to(caches[0].device), *caches), dim=0)

    def read_cache(self, cache=None):
        # read values from cache and convert to correct types
        mention_spans = (self.cache if cache is None else cache)[:, 0, :, :self.max_mention_span].long()
        candidate_ids = (self.cache if cache is None else cache)[:, 1, :, :self.max_candidates].long()
        candidate_mask = (self.cache if cache is None else cache)[:, 2, :, :self.max_candidates].bool()
        candidate_priors = (self.cache if cache is None else cache)[:, 3, :, :self.max_candidates].float()
        # clear cache after reading to prevent 
        # false double use of the same cache
        self.clear_cache()
        # return values
        return mention_spans, candidate_ids, candidate_mask, candidate_priors

    def forward(self, h, cache=None):
        
        # read cache and move all to device
        mention_spans, candidate_ids, candidate_mask, candidate_priors = self.read_cache(cache)
        mention_spans, candidate_ids, candidate_mask, candidate_priors = mention_spans.to(h.device), candidate_ids.to(h.device), candidate_mask.to(h.device), candidate_priors.to(h.device)
        # check cache values
        if (len(mention_spans) != h.size(0)) or (candidate_ids.size(0) != h.size(0)) \
                or (candidate_mask.size(0) != h.size(0)) or (candidate_priors.size(0) != h.size(0)):
            raise RuntimeError("Cache size (%i) does not match batch-size (%i)!" % (len(mention_spans), h.size(0)))

        # get candidate-embeddings and build mention-mask
        candidate_embs = self.kb.embedd(candidate_ids).detach().to(h.device) # no gradients for entity embeddings
        mention_mask = ((mention_spans >= 0).sum(-1) > 0)

        # project hidden into entity-embedding space
        h_proj = self.bert2kb(h)
        # compute entity linking scores
        mention_span_reprs = self.mention_span_representer(h_proj, mention_spans)
        linking_scores = self.entity_linker(candidate_embs, candidate_mask, candidate_priors, mention_span_reprs, mention_mask)
        enhanced_span_reprs, entropy = self.enhanced_representation(linking_scores, candidate_embs, mention_span_reprs, mention_mask)
        recontextualized_reprs = self.recontextualizer(enhanced_span_reprs, h_proj, mention_mask)
        # project from knowledge-base back to bert
        h_new = self.dropout(self.kb2bert(recontextualized_reprs))
        h_new = self.output_ln(h + h_new)

        # return new hidden state
        return h_new, linking_scores, entropy
