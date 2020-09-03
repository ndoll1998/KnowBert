from transformers import BertConfig
from .knowledge import KnowledgeBase

class KnowBertConfig(BertConfig):

    def __init__(self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        kbs={},
        pretrained_model_name_or_path=None,
        **kwargs
    ):
        # initialize bert config
        BertConfig.__init__(self,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )
        # save knowledge configurations
        self.kbs = kbs
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def add_kb(self, layer:int, kb:KnowledgeBase, kar_kwargs:dict) -> None:
        # check if kb is of correct type
        if not isinstance(kb, KnowledgeBase):
            raise RuntimeError("%s must inherit KnowledgeBase" % kb.__class__.__name__)
        # check if layer already has a kb
        if self.kbs.get(layer, None) is not None:
            raise RuntimeError("There already is a knowledge base at layer %i" % layer)

        # build full config
        config = kb.config
        config.update({'kar_kwargs': kar_kwargs})
        # add knowledge base to config
        self.kbs[layer] = config

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        # create configuration from model name or path
        config_dict, kwargs = KnowBertConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        kwargs.update({'pretrained_model_name_or_path': pretrained_model_name_or_path})
        return KnowBertConfig.from_dict(config_dict, **kwargs)
