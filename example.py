import torch
import transformers
from kb.model import KnowBert, KnowBertForPretraining
from kb.knowledge import KnowledgeBase

class TestKB(KnowledgeBase):
    """ Simple af knowledge base """

    def __init__(self):
        super(TestKB, self).__init__(embedd_dim=12)

    def find_mentions(self, tokens):
        # hard coded mentions for both examples
        if tokens[0].lower() == "this":
            return {"nice_coffee": [3, 4]}
        if tokens[0].lower() == "i":
            return {"like": [3], "place": [5], "very_much": [6, 7]}

    def find_candidates(self, mention):
        # mentions are entity candidates
        # different number of candidates for different mentions
        if mention == "like":
            return [1, 1]
        return [1]

    def get_prior(self, entity_id):
        return 1

    def embedd(self, entity_ids):
        # random entity embedding
        return torch.empty((*entity_ids.size(), self.embedd_dim)).uniform_(-1, 1)

# sample sentences
sampleA = "This is a nice coffee spot and the food was tasty too!"
sampleB = "I do not like this place very much!"

# create bert model
bert = KnowBert.from_pretrained("bert-base-uncased")
# add knowledge base
kb_A = bert.add_kb(3, TestKB())
kb_B = bert.add_kb(2, TestKB())

# create tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
tokensA = tokenizer.tokenize(sampleA)
tokensB = tokenizer.tokenize(sampleB)
l = max(len(tokensA), len(tokensB))
input_ids_A = tokenizer.convert_tokens_to_ids(tokensA) + [tokenizer.sep_token_id] * (l - len(tokensA))
input_ids_B = tokenizer.convert_tokens_to_ids(tokensB) + [tokenizer.sep_token_id] * (l - len(tokensB))
# create tensor
input_ids = torch.tensor([input_ids_A, input_ids_B]).long()

# prepare and execute
bert.prepare_kbs([tokensA, tokensB])
output = bert.forward(input_ids)
