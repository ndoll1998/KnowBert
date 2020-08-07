import torch
import torch.nn as nn
import transformers
# import knowbert model
from kb.model import KnowBertForPretraining
# import knowledge bases
from senticnet.kb import SenticNet
# utils
import os
from time import time
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

def load_data(data_path, batch_size):
    # load data
    data = torch.load(data_path, map_location='cpu')
    data, caches = data['data'], data['caches']
    # separate data into training and testing dataset
    train_length = int(0.9 * data[0].size(0))
    train_data = (t[:train_length] for t in data)
    test_data = (t[train_length:] for t in data)
    # separate caches
    train_caches = [[cached[:train_length, ...] for cached in cache] for cache in caches]
    test_caches = [[cached[train_length:, ...] for cached in cache] for cache in caches]

    # create datasets
    train_dataset = torch.utils.data.TensorDataset(*train_data, *sum(train_caches, []))
    test_dataset = torch.utils.data.TensorDataset(*test_data, *sum(test_caches, []))
    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    return train_dataloader, test_dataloader


def set_caches(model, cached):
    # get layers wich have a knowledge base added
    kb_layers = [i for i, kb in enumerate(model.kbs) if kb is not None]
    n_kbs = len(kb_layers)

    # create caches
    only_valid_caches = [{
            'mention_spans': cached[i + 0],
            'candidate_ids': cached[i + 1],
            'candidate_mask': cached[i + 2],
            'candidate_priors': cached[i + 3]
        } for i in range(0, n_kbs * 4, 4)
    ]
    # expand caches - add Nones
    caches = [None] * len(model.kbs)
    for l, cache in zip(kb_layers, only_valid_caches):
        caches[l] = cache
    
    # set caches
    model.reset_kb_caches()
    model.stack_kb_caches(caches)


def predict(model, batch, device):
    # separate data and cached
    data, cached = batch[:4], batch[4:]
    # unpack data and update caches
    input_ids, token_type_ids, next_sentence_labels, masked_labels = data
    set_caches(model.module if type(model) is nn.DataParallel else model, cached)
    # predict
    return model.forward(
        input_ids=input_ids.to(device), 
        token_type_ids=token_type_ids.to(device), 
        labels=masked_labels.to(device),
        next_sentence_label=next_sentence_labels.to(device)
    )


if __name__ == '__main__':

    main_device = 'cuda:0'
    # base model and data path
    bert_base_model = "bert-base-uncased"
    data_path = "data/pretraining_data/english_yelp_reviews.pkl"
    dump_path = "data/results/bert-base-uncased"
    # optimizer and data preparation
    epochs = 20
    batch_size = 64

    # create dump folder
    os.makedirs(dump_path, exist_ok=True)

    print("Loading Model... (%s)" % bert_base_model)

    # create model and add knowledge bases
    # make sure the order of the knowledge bases is the same when preprocessing the data
    # the exact positions of the kbs can very as long as the order is kept
    model = KnowBertForPretraining.from_pretrained(bert_base_model)
    kb = model.add_kb(10, SenticNet(mode="train"))
    # only compute gradients down to layer 10
    model.freeze_layers(10)
    
    # if torch.cuda.device_count() > 1:
        # use all devices
        # model = nn.DataParallel(model)
    # use main-device
    model.to(main_device)


    print("Creating Optimizer...")

    # create optimizer
    optim = torch.optim.Adam(kb.parameters())   # only train knowledge base
    # optim = torch.optim.Adam(model.parameters())  # train all unfrozen parameters


    print("Loading Data...")

    # create train and test-dataloader
    train_dataloader, test_dataloader = load_data(data_path, batch_size)


    print("Starting Training...")

    # training loop
    train_losses, test_losses = [], []
    all_mask_lm_f1_scores, all_next_sentence_f1_scores = [], []
    for e in range(1, 1 + epochs):

        print("Epoch %i" % e)
        st = time()

        model.train()
        running_loss = 0
        # train model
        for i, batch in enumerate(train_dataloader, start=1):
            # predict and get output
            loss, _, _ = predict(model, batch, main_device)
            # backpropagate and update parameters
            optim.zero_grad()
            loss.backward()
            optim.step()
            # log process
            running_loss += loss.item()
            print("\tStep %i/%i\t - Train-Loss %.4f\t - Time %.4fs" % (i, len(train_dataloader), running_loss / i, time() - st), end="\r")
    
        train_losses.append(running_loss / len(train_dataloader))

        model.eval()
        running_loss = 0
        # test model
        all_mask_lm_preds, all_next_sentence_preds = [], []
        all_mask_lm_targets, all_next_sentence_targets = [], []

        for batch in test_dataloader:
            # predict
            loss, mask_lm_scores, next_sentence_scores = predict(model, batch, main_device)
            running_loss += loss.item()
            # get predictions from scores
            mask_lm_preds = mask_lm_scores.max(dim=-1)[1].cpu().flatten()
            next_sentence_preds = next_sentence_scores.max(dim=1)[1].cpu().flatten()
            # get target values from batch
            mask_lm_targets, next_sentence_targets = batch[3].flatten(), batch[2].flatten()

            # create and apply mask for lm prediction
            mask_lm_mask = (mask_lm_targets != -100)
            mask_lm_preds = mask_lm_preds[mask_lm_mask]
            mask_lm_targets = mask_lm_targets[mask_lm_mask]

            # extend lists
            all_mask_lm_preds.extend(mask_lm_preds.tolist())
            all_mask_lm_targets.extend(mask_lm_targets.tolist())
            all_next_sentence_preds.extend(next_sentence_preds.tolist())
            all_next_sentence_targets.extend(next_sentence_targets.tolist())
        
        # compute f1-scores
        mask_lm_f1_score = f1_score(all_mask_lm_targets, all_mask_lm_preds, average='micro')
        next_sentence_f1_score = f1_score(all_next_sentence_targets, all_next_sentence_preds)

        # add to lists
        all_mask_lm_f1_scores.append(mask_lm_f1_score)
        all_next_sentence_f1_scores.append(next_sentence_f1_score)
        test_losses.append(running_loss / len(test_dataloader))

        print("\n\tTest-Loss %.4f\t - Mask-LM-F1 %.4f\t - Next-Sent-F1 %.4f" % (test_losses[-1], mask_lm_f1_score, next_sentence_f1_score))

        # save model parameters
        torch.save(model.state_dict(), os.path.join(dump_path, "model-E%i.pkl" % e))

    print("Saving results...")
    # save final results
    with open(os.path.join(dump_path, "results.txt"), "w+") as f:
        f.write("Train-Loss:\t %f" % train_losses[-1])
        f.write("Test-Loss:\t %f" % test_losses[-1])
        f.write("Mask-LM-F1:\t %f" % all_mask_lm_f1_scores[-1])
        f.write("Next-Sentence-F1:\t %f" % all_next_sentence_f1_scores[-1])
    # create and save plot for losses
    fig, ax = plt.subplots(1, 1)
    ax.plot(train_losses)
    ax.plot(test_losses)
    ax.legend(["Train", "Test"])
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training and Testing Loss over Epochs")
    fig.savefig(os.path.join(dump_path, "losses.png"), format='png')
    # create and save plot for f1-scores
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_mask_lm_f1_scores)
    ax.plot(all_next_sentence_f1_scores)
    ax.set(xlabel="Epoch", ylabel="F1-Scores", title="F1-Scores for Mask-LM and Next-Sentence")

    # save model parameters
    torch.save(model.state_dict(), os.path.join(dump_path, "model-Final.pkl"))
    # save optimizer parameters for further training
    torch.save(optim.state_dict(), os.path.join(dump_path, "optimizer.pkl"))
