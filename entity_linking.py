import os
import torch
import transformers
from kb.model import KnowBertForPretraining
from knowledgeBases.senticnet.kb import SenticNet

# sample text
sample = "Die Atmosphäre ist auch sehr angenehm, trotz eines Samstag Abends fanden wir direkt einen Platz & die Lautstärke ist im oberen Stockwerk im toleranten Bereich."
# model
tokenizer = "bert-base-german-cased"
bert_base_model = "pretrained/bert-base-german-cased-yelp-entropy"


# create tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer)
# create model
model = KnowBertForPretraining.from_pretrained(bert_base_model)
model.eval()

# tokenize sample
tokens = tokenizer.tokenize(sample)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor([input_ids])

# prepare and apply model
mention_candidates = model.prepare_kbs([tokens])[0]
linking_scores = model.forward(input_ids=input_ids)[-2]

for layer, (layer_scores, d) in enumerate(zip(linking_scores, mention_candidates)):  
    if d is not None:
        for (term, candidate_ids), scores in zip(d, layer_scores[0]):
            # get valid candidates and scores
            candidate_mask = (candidate_ids != -1)
            candidate_ids, scores = candidate_ids[candidate_mask], torch.softmax(scores[candidate_mask], dim=0)
            # print term and entropy
            print("Term: %s" % term)
            print("Entropy: %f" % -(scores * torch.log(scores)).sum().item())
            candidate_ids, scores = candidate_ids.tolist(), scores.tolist()
            # print candidates with their scores
            for candidate_id, score in zip(candidate_ids, scores):
                candidate_term = model.kbs[layer].kb.id2entity(candidate_id)
                print(candidate_term, score)

            print()
