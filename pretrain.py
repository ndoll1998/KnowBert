import torch
import torch.nn as nn
import transformers
# import knowbert model
from kb.model import KnowBertForPretraining
# import knowledge bases
from senticnet.kb import SenticNet
# utils
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

def load_data(data_path, batch_size):
   
    # find all files that match pattern
    all_files = glob.glob(data_path)[:1000]

    data_tensors, all_caches = [], []
    # load all data files
    for i, fpath in enumerate(all_files, 1):
        # load current file
        data = torch.load(fpath, map_location='cpu')
        data, caches = data['data'], data['caches']
        data_tensors.append(data)
        all_caches.append(caches)
        # log
        print("(%i/%i) Loaded %i items from %s" % (i, len(all_files), data[0].size(0), fpath))

    # concatenate all tensors from the different files
    data = [torch.cat(tensors, dim=0) for tensors in zip(*data_tensors)]
    caches = [torch.cat(caches, dim=0) for caches in zip(*all_caches)]

    # training data portion
    train_length = int(0.9 * data[0].size(0))
    print("Using %i out of %i items for training" % (train_length, data[0].size(0)))

    # separating data into training and testing data
    train_data = (t[:train_length] for t in data)
    test_data = (t[train_length:] for t in data)
    # separate caches
    train_caches = [cache[:train_length] for cache in caches]
    test_caches = [cache[train_length:] for cache in caches]

    # create datasets
    train_dataset = torch.utils.data.TensorDataset(*train_data, *train_caches)
    test_dataset = torch.utils.data.TensorDataset(*test_data, *test_caches)
    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

def predict(model, batch, device):
    # separate data and cached
    data, caches = batch[:4], batch[4:]
    input_ids, token_type_ids, next_sentence_labels, masked_labels = data
    # predict
    return model.forward(
        *caches,
        input_ids=input_ids.to(device), 
        token_type_ids=token_type_ids.to(device), 
        labels=masked_labels.to(device),
        next_sentence_label=next_sentence_labels.to(device)
    )


if __name__ == '__main__':

    main_device = 'cuda:0'
    # base model and data path
    bert_base_model = "bert-base-uncased"
    data_path = "data/pretraining_data/english_wiki/processed/*.pkl"
    dump_path = "data/results/bert-base-uncased-wiki"
    # optimizer and data preparation
    epochs = 5
    batch_size = 64 * torch.cuda.device_count()

    # create dump folder
    os.makedirs(dump_path, exist_ok=True)

    print("Loading Model... (%s)" % bert_base_model)

    # create model and add knowledge bases
    # make sure the order of the knowledge bases is the same when preprocessing the data
    # the exact positions of the kbs can very as long as the order is kept
    model = KnowBertForPretraining.from_pretrained(bert_base_model)
    kb = model.add_kb(10, SenticNet(data_path="data/senticnet/english", mode="train"))
    # only compute gradients down to layer 10
    # model.freeze_layers(10)
    
    
    if torch.cuda.device_count() > 1:
        print("Using %i cuda devices!" % torch.cuda.device_count())
        model = nn.DataParallel(model, output_device=main_device)
    # use all devices
    model.to(main_device)

    print("Creating Optimizer...")

    # create optimizer
    # optim = torch.optim.Adam(kb.parameters())   # only train knowledge base
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)  # train all unfrozen parameters


    print("Loading Data...")

    # create train and test-dataloader
    train_dataloader, test_dataloader = load_data(data_path, batch_size)


    print("Starting Training...")

    # training loop
    train_losses, test_losses = [], []
    all_mask_lm_f1_scores, all_next_sentence_f1_scores = [], []
    for e in range(1, 1 + epochs):

        print("Epoch %i" % e)

        with tqdm(total=len(train_dataloader)) as progress:
            progress.set_description("Train")

            model.train()
            running_loss = 0
            # train model
            for i, batch in enumerate(train_dataloader, start=1):
                # predict and get output
                loss, _, _, _ = predict(model, batch, main_device)
                loss = loss.mean()
                # backpropagate and update parameters
                optim.zero_grad()
                loss.backward()
                optim.step()
                # log process
                running_loss += loss.item()
        
                # update progress bar
                progress.update(1)
                progress.set_postfix({'loss': running_loss / i}, refresh=True)

    
                if i % 10000 == 0:
                    # save checkpoint every 10_000 steps
                    torch.save(model.state_dict(), os.path.join(dump_path, "model-ckpt-%i-%i.pkl" % (e, i)))
        
            train_losses.append(running_loss / len(train_dataloader))
        
        with tqdm(total=len(test_dataloader)) as progress:
            progress.set_description("Test")

            model.eval()
            running_loss = 0
            # test model
            all_mask_lm_preds, all_next_sentence_preds = [], []
            all_mask_lm_targets, all_next_sentence_targets = [], []
        
            with torch.no_grad():
                for i, batch in enumerate(test_dataloader, 1):
                    # predict
                    loss, mask_lm_scores, next_sentence_scores, _ = predict(model, batch, main_device)
                    running_loss += loss.mean().item()
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
                    
                    # update progress bar
                    progress.update(1)
                    progress.set_postfix({'loss': running_loss / i}, refresh=True)

        # compute f1-scores
        mask_lm_f1_score = f1_score(all_mask_lm_targets, all_mask_lm_preds, average='micro')
        next_sentence_f1_score = f1_score(all_next_sentence_targets, all_next_sentence_preds)

        # add to lists
        all_mask_lm_f1_scores.append(mask_lm_f1_score)
        all_next_sentence_f1_scores.append(next_sentence_f1_score)
        test_losses.append(running_loss / len(test_dataloader))

        print("Epoch %i - Mask-LM-F1 %.4f\t - Next-Sent-F1 %.4f" % (e, mask_lm_f1_score, next_sentence_f1_score))

        # save model parameters
        torch.save(model.state_dict(), os.path.join(dump_path, "model-ckpt-%i.pkl" % e))
    
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
    module = model.module if type(model) is nn.DataParallel else model
    module.save_pretrained(dump_path)
    # save optimizer parameters for further training
    torch.save(optim.state_dict(), os.path.join(dump_path, "optimizer.pkl"))
