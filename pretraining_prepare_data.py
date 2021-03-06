import torch
import random
from tqdm import tqdm

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_training_data(
    doc_id, all_documents, 
    knowbert_model, tokenizer,
    max_seq_length, max_predictions_per_seq,
    short_seq_prob, masked_lm_prob, 
    whole_word_mask
):
    # get tokenizer vocab
    vocab = list(tokenizer.vocab.keys())

    # get current document
    doc = all_documents[doc_id]
    # account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    # use short sequences with some probability
    target_seq_length = max_seq_length if random.random() >= short_seq_prob else random.randint(2, max_num_tokens)
    
    # resulting lists
    all_token_ids = []
    all_segment_ids = []
    all_random_nexts = []
    all_masked_lm_labels = []

    i = 0
    current_chunk = []
    current_length = 0
    while i < len(doc):

        segment = doc[i]
        current_chunk.append(segment)
        current_length += len(segment)

        # fill current chunk until target sequence length is reached
        if i == len(doc) - 1 or current_length >= target_seq_length:

            if len(current_chunk) <= 1:
                continue
            
            # build first (A) segment/sentence
            segments_a = 1 if len(current_chunk) == 2 else random.randint(1, len(current_chunk) - 2)
            tokens_a = sum([doc[j] for j in range(segments_a)], [])

            # build second (B) segment/sentence
            tokens_b = []
            # random next - for next sentence prediction
            is_random_next = (len(current_chunk) == 1) or (random.random() < 0.5)
            is_random_next = is_random_next and (len(all_documents) > 1)

            if is_random_next:
                target_b_length = target_seq_length - len(tokens_a)
                # choose a different random document
                doc_id_pool = list(range(0, doc_id)) + list(range(doc_id+1, len(all_documents)))
                random_doc_id = random.choice(doc_id_pool)
                random_doc = all_documents[random_doc_id]
                # choose a random start and fill segment until target sequence length is reached
                start = random.randint(0, len(random_doc) - 1)
                for j in range(start, len(random_doc)):
                    tokens_b.extend(random_doc[j])
                    if len(tokens_b) > target_b_length:
                        break

                # re-iterate tokens that were not used 
                i -= len(current_chunk) - segments_a

            else:
                # add all remaining tokens to sentence B
                tokens_b.extend(sum(current_chunk[segments_a:], []))

            # truncate to keep sequence length
            truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

            assert len(tokens_a) >= 1
            assert len(tokens_b) >= 1

            # build token-sequence from both token-lists
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            # create and stack all kb-caches for current tokens
            knowbert_model.stack_kb_caches(knowbert_model.build_kb_caches(tokens))

            # create masked lm data
            masked_tokens, masked_idx, masked_targets = create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, whole_word_mask)
            
            # convert tokens to ids
            token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
            masked_target_ids = tokenizer.convert_tokens_to_ids(masked_targets)
            # create masked target labels
            masked_target_labels = torch.zeros(max_seq_length).long() - 100
            masked_target_labels[masked_idx] = torch.LongTensor(masked_target_ids)

            # fill
            token_ids += [tokenizer.pad_token_id] * (max_seq_length - len(token_ids))
            segment_ids += [0] * (max_seq_length - len(segment_ids))
            # convert to tensors and add to corresponding list
            all_random_nexts.append(is_random_next)
            all_token_ids.append(torch.LongTensor(token_ids))
            all_segment_ids.append(torch.LongTensor(segment_ids))
            all_masked_lm_labels.append(masked_target_labels)

            # reset chunk
            current_chunk = []
            current_length = 0

        # increase counter
        i += 1

    # return everything
    return all_random_nexts, all_token_ids, all_segment_ids, all_masked_lm_labels

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, whole_word_mask):

    # get all cadidate indices
    candidate_idx = []
    for i, token in enumerate(tokens):
        # ignore special tokens
        if token in ['[CLS]', '[SEP]']:
            continue
        # add index to candidate list
        if whole_word_mask and (len(candidate_idx) > 0) and token.startswith('##'):
            candidate_idx[-1].append(i)
        else:
            candidate_idx.append([i])

    # shuffle
    random.shuffle(candidate_idx)
    # number of predictions
    n_preds = min(max_predictions_per_seq, max(1, round(len(tokens) * masked_lm_prob)))

    masked_idx, masked_targets = [], []
    for idx in candidate_idx:
        # enough tokens selected
        if len(masked_targets) >= n_preds:
            break
        # maximum number would be exceeded
        if len(masked_targets) + len(idx) > n_preds:
            continue

        for i in idx:
            masked_idx.append(i)
            masked_targets.append(tokens[i])
            # choose replacement for token
            rand = random.random()
            if rand < 0.1:
                # keep token in 10% of the times
                pass
            elif rand < 0.2:
                # replace it with a random word with prob of 10%
                tokens[i] = vocab[random.randint(0, len(vocab)-1)]
            elif rand < 1:
                # 80% of the times mask the token
                tokens[i] = "[MASK]"

    return tokens, masked_idx, masked_targets


if __name__ == '__main__':

    import os
    from glob import glob
    # import model and tokenizer
    from src.kb.model import KnowBertModel, BertConfig
    from transformers import BertTokenizer
    # import knowledge bases
    from src.knowledge.senticnet.kb import SenticNet

    bert_base_model = "bert-base-uncased"
    source_file_path = "data/pretraining_data/english_wiki/txt/*"
    output_path = "data/pretraining_data/english_wiki/processed"
    # create dump-path
    os.makedirs(output_path, exist_ok=True)

    kwargs = {
        'max_seq_length': 128,
        'short_seq_prob': 0.1,
        'masked_lm_prob': 0.15,
        'max_predictions_per_seq': 20,
        'whole_word_mask': True
    }

    # create model - only needed to create caches for the knowledge bases
    config = BertConfig.from_pretrained(bert_base_model)
    model = KnowBertModel(config)
    # add knowledge bases
    model.add_kb(10, SenticNet(data_path='./data/senticnet/english'))

    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_base_model)
    tokenize = lambda text: tokenizer.tokenize(text.strip())

    for fpath in glob(source_file_path):

        print("Reading and Tokenizing Documents...(%s)" % fpath)

        # read documents from file
        with open(fpath, 'r', encoding='utf-8') as f:
            documents = f.read().split('\n\n')
            documents = [[tokenize(sent) for sent in doc.strip().split('\n')] for doc in documents if len(doc.strip()) > 0]
        # shuffle documents
        random.shuffle(documents)

        print("Preprocessing Documents...")

        # reset caches
        model.clear_kb_caches()
        # features to extract
        is_random_nexts, token_ids, segment_ids, masked_labels = [], [], [], []
        # create training data from each document separately
        for i in tqdm(range(len(documents))):
            # get features - this also build the corresponfing caches in the model
            features = create_training_data(i, documents, model, tokenizer, **kwargs)
            # add features to lists
            is_random_nexts.extend(features[0])
            token_ids.extend(features[1])
            segment_ids.extend(features[2])
            masked_labels.extend(features[3])

        # get caches from model
        caches = [cache for cache in model.get_kb_caches() if cache is not None]
        # stack feature tensors    
        data = (
            torch.stack(token_ids, dim=0), 
            torch.stack(segment_ids, dim=0),
            torch.tensor(is_random_nexts).long(), 
            torch.stack(masked_labels, dim=0),
        )

        print("Saving Preprocessed Data...")

        # build output file path
        _, fname = os.path.split(fpath)
        output_fpath = os.path.join(output_path, fname.replace('.txt', '.pkl'))
        output_fpath = output_fpath if '.pkl' in output_fpath else (output_fpath + '.pkl')
        # save tensors to output-file
        torch.save({'data': data, 'caches': caches}, output_fpath)

        print("Saved Training Data to %s" % output_fpath)

    print("Finished Preprocessing!")
