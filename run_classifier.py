"""
Main classification pipeline for semantic pointers.
"""
#西田拳


from __future__ import print_function, division


import pprint as pp
import numpy as np
import time
import os
import copy
import argparse
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd
import wandb
import random
from datetime import datetime
import math

now = datetime.now()

# Torch functions.
import torch
import torch.optim as optim
from torch.utils import data
from torch.optim import lr_scheduler

#from lib.tf_idf_data import SPNDataset, MultiLabelDataset, MultiLabelSparseDataset
from lib.data import SPNDataset, MultiLabelDataset, MultiLabelSparseDataset
from lib.data import SPNSparseDataset, Sampler
#from lib.tf_idf_data import SPNSparseDataset, Sampler
from lib.model import SemanticPointerNetwork,SemanticPointerNetworkHalf,SemanticPointerNetworkTanh, SemanticPointerNetworkSin
from lib.model import spp_train, spp_test
from lib.metrics import plot_tr_stats, compute_inv_propensity, display_metrics, display_agg_results, compute_Nl_dict
from lib.utils import print_memory_profile, ExperimentTime, print_command_arguments

# Commandline Arguments.
parser = argparse.ArgumentParser(description='Extreme Multi-label Classification.')
parser.add_argument('--name', type=str, default='test', help='A unique experiment name.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 250)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--th', type=float, default=0.0, help='Theshold for label inference.')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=200, metavar='S',
                    help='random seed (default: 100)')
parser.add_argument('--topk', type=int, default=5, metavar='S',
                    help='Retreive top k labels (Default: 5).')
parser.add_argument('-a', type=float, default=0.55,
                    help='Inverse propensity value A (Default: 0.55).')
parser.add_argument('-b', type=float, default=1.5,
                    help='Inverse propensity value B (Default: 1.5).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', type=str, default='.', help='Directory to save model and results.')
parser.add_argument('--data-file', type=str, default="None", help='Location of the data CSV file.')
parser.add_argument('--tr-split', type=str, help='Get the training split for dataset.')
parser.add_argument('--te-split', type=str, help='Get the test split for dataset.')
parser.add_argument('--test', action='store_true', default=False,
                    help='Perform tests on pretrained model.')
parser.add_argument('--reduce-dims', action='store_true', default=False,
                    help='Reduce dimensions of the features.')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Print debug statements for verification.')
parser.add_argument('--use_fc3', action='store_true', default=False,
                    help='Print debug statements for verification.')
parser.add_argument('--proloss', action='store_true', default=False,
                    help='Print debug statements for verification.')
parser.add_argument('--h-size', type=int, default= 768,
                    help='Setting hidden size h.')

# SPN specific arguments.
parser.add_argument('--baseline', action='store_true', default=False,
                    help='Use Baseline Network.')
parser.add_argument('--spn-dim', type=int, default=400, metavar='S',
                    help='Label vector dimensions (Default: 400)')
parser.add_argument('--no-grad', action='store_true', default=False,
                    help='Update Label vectors.')
parser.add_argument('--without-negative', action='store_true', default=False,
                    help='disable negative loss.')
parser.add_argument('--load-vec', type=str, default=None, help='Location of pretrained vectors.')
parser.add_argument('--normalize', type=str, default=True, help='normalize')
parser.add_argument('--npseed', type=int, default=200, metavar='S',
                    help='random seed (default: 100)')
parser.add_argument('--first', action='store_true', default=False)
parser.add_argument('--model-type', type=str, default='default',
                    choices=['default', 'half', 'tanh', 'sin'],
                    help='Type of SemanticPointerNetwork model to use. Choices: default, half, tanh, sin.')


# Start.
args = parser.parse_args()
print_command_arguments(args)

###################################################
###################################################
###################################################
project_name = "2024_01_26"
####################################################
####################################################
####################################################

wandb.init(project = project_name , config = args)

# Locations.
model_path = args.save + "/" + args.name + ".pth"

# Begin.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.npseed)
device = torch.device("cuda" if use_cuda else "cpu")
#kwargs = {'num_workers': 16, 'pin_memory': True, 'drop_last': True} if use_cuda else {'drop_last': True, 'num_workers': 8}
kwargs = {'num_workers': 2, 'prefetch_factor' : 2, 'pin_memory': True, 'drop_last': True} if use_cuda else {'drop_last': True, 'num_workers': 2}

# Device.
print("Device Used: {}".format(device))

# Create the dataset.
if args.data_file != "None":
    if args.baseline is True:
            eld = MultiLabelDataset(args.data_file,
                                    dimension_reduction=args.reduce_dims)
    else:
            eld = SPNDataset(args.data_file,
                             dimension_reduction=args.reduce_dims)
            max_pred_size = eld.get_max_size()

    num_classes = eld.num_labels # Number of labels in the datasets.
    num_features = len(eld.features[0])

    # Create the sampler object that contains the splits for training and test.
    sampler_idx = 0 # Idx of the sampler selected for testing.
    eld_sampler = Sampler(args.tr_split, args.te_split)
    train_sampler, validation_sampler, test_sampler = eld_sampler[sampler_idx]

    # Dataloaders for training, validation and testing.
    train_loader = torch.utils.data.DataLoader(eld, batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(eld, batch_size=args.test_batch_size,
                                             sampler=validation_sampler,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(eld, batch_size=args.test_batch_size,
                                              sampler=test_sampler,
                                              **kwargs)

else: # When a single train split are available.
    if args.baseline is True:
            eld_train = MultiLabelSparseDataset(args.tr_split)
            eld_test = MultiLabelSparseDataset(args.te_split)
    else:
            eld_train = SPNSparseDataset(args.tr_split)
            eld_test = SPNSparseDataset(args.te_split)
            max_pred_size = eld_train.get_max_size()

    num_classes = eld_train.num_labels # Number of labels in the datasets.
    num_features = eld_train.features.shape[1]

    # Split the training data into training and validation.
    n_train = int(len(eld_train) * 0.9)
    n_val = len(eld_train) - n_train
    train_dataset, val_dataset = data.random_split(eld_train, (n_train, n_val), generator=torch.Generator().manual_seed(0))


    print("Training dataset: {}, Validation dataset: {}".format(len(train_dataset),
                                                                len(val_dataset)))

    # Dataloaders for training, validation and testing.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                              shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(eld_test, batch_size=args.test_batch_size,
                                              shuffle=True, **kwargs)
    

# Compute propensity based on training labels.
# NOTE: http://manikvarma.org/downloads/XC/XMLRepository.html#Jain16
# http://manikvarma.org/pubs/jain16.pdf
if args.data_file != "None":
    if args.baseline is True:
        labels = train_loader.dataset.labels[eld_sampler.train_idx[sampler_idx]]
    else:
        labels = train_loader.dataset.get_one_hot(eld_sampler.train_idx[sampler_idx])

    inv_propen = compute_inv_propensity(labels, A=args.a, B=args.b)

    if args.debug is True:
        pp.pprint("--------Sample----------")
        pp.pprint(eld.features[10].shape)
        pp.pprint(eld.labels[10].shape)
        pp.pprint("--------Sample----------")
else:
    if args.baseline is True:
        labels = train_loader.dataset.dataset.labels
    else:
        labels = train_loader.dataset.dataset.splabels

    inv_propen = compute_inv_propensity(labels, A=args.a, B=args.b)

    if args.debug is True:
        pp.pprint("--------Sample----------")
        pp.pprint(eld_train.features[10].shape)
        pp.pprint(eld_train.labels[10].shape)
        pp.pprint("--------Sample----------")


# Nl_dict = compute_Nl_dict(labels)
# W_dict = {}
# C = (math.log(n_train)-1) * (args.b + 1) ** args.a

# for l in Nl_dict:
#     Nl = Nl_dict[l]
#     W_dict[l] = 1 + C * (Nl + args.b) ** -args.a

Nl_dict = {}
spn_dims = [1500,3000,4000,5000,7500,10000]


for spn_dim in spn_dims:
    if args.test is not True:
        # Start Experiment.
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        if args.baseline is True:
            exit()
        else:
            if args.debug:
                print("Create the semantic pointer network...")

            if args.model_type == 'default':
                model_cls = SemanticPointerNetwork
            elif args.model_type == 'half':
                model_cls = SemanticPointerNetworkHalf
            elif args.model_type == 'tanh':
                model_cls = SemanticPointerNetworkTanh
            elif args.model_type == 'sin':
                model_cls = SemanticPointerNetworkSin
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")
            

            model = model_cls(num_features, num_classes, spn_dim, args.h_size,Nl_dict,
                              max_pred_size=max_pred_size, A= args.a, B=args.b,
                              no_grad=args.no_grad, load_vec=args.load_vec,
                              debug=args.debug, use_fc3=args.use_fc3).to(device)
            

            # Create embedding layer for CPU.
            model.create_label_embedding()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

        # Training Stats.
        # Optimize model on validation data.
        # Strategy: Save the model with the least validation loss and load for inference during tests.
        tr_stats = []; spoch = 0
        min_val_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
        #for epoch in range(1, 4):
            print("Training Epochs: {}".format(epoch))
            if args.baseline is True:
                exit()
            else:
                if args.proloss is True:
                    tr_loss, _, _ = spp_train(args.log_interval, model, device,
                                        train_loader, optimizer, epoch, without_negative=args.without_negative, proloss=args.proloss, Nl_dict= Nl_dict, N=num_classes, A=args.a, B=args.b)
                    val_loss, f1 ,pr, rec = spp_test(model, device, val_loader,
                                                topk=args.topk, without_negative=args.without_negative, proloss=args.proloss, Nl_dict= Nl_dict, N=num_classes, A=args.a, B=args.b)
                else:
                    tr_loss, _, _ = spp_train(args.log_interval, model, device,
                                            train_loader, optimizer, epoch, without_negative=args.without_negative)
                    val_loss, f1 ,pr, rec = spp_test(model, device, val_loader,
                                                    topk=args.topk, without_negative=args.without_negative)

            if min_val_loss > val_loss:
                min_val_loss = val_loss

                # Save the current model.
                if args.debug:
                    print("Saving model @ Epoch: {}".format(epoch))

                torch.save(model, model_path)
                spoch = epoch

            scheduler.step()
            tr_stats.append([epoch, tr_loss, val_loss, pr, rec, f1])

        # Save the final model too.
        torch.save(model, model_path + ".fin") # Final model.

        # Stats.
        tr_stats = pd.DataFrame(tr_stats)
        tr_stats.columns = ['Epoch', 'Training Loss', 'Val Loss', 'Precision',
                            'Recall', 'F1 Score']

        # Perform grid search for the threshold.
        # NOTE: Select optimal threshold based on F-1 score.
        th_stats = []; max_f1 = 0.0; optimal = []; th = 0.05
        # Skip
        """
        while th < 1.0:
            if args.baseline is True:
            exit()
            else:
                # For SPP.
                _, f1, pr, rec = spp_test(model, device, test_loader, threshold=th,
                                        without_negative=args.without_negative)

            row = [th, pr, rec, f1]
            th_stats.append(row)

            if max_f1 < f1:
                max_f1 = f1
                optimal = row

            th += 0.05
        """

        th_stats = pd.DataFrame(th_stats)
        #th_stats.columns = ['Threshold', 'Precision', 'Recall', 'F1 Score']

        #args.th = optimal[0] # Optimal threshold.

        # Print results.
        if args.debug:
            print(tabulate(tr_stats, headers=["Epoch", "TR Loss","Validation Loss",
                                            "Validation Precision",
                                            "Validation Recall",
                                            "Validation F-1 Score"]))

            print(tabulate(tr_stats, headers=["Threshold", "Precision", "Recall",
                                            "F-1 Score"]))

        #plot_tr_stats(tr_stats, th_stats, spoch, optimal[0], args.save + "/" + args.name + ".tr.fig")

        print("-----------Running Measurements---------")
        print("Training time / epoch: {:0.3f}".format(model.time['train'].get_elapsed_time() / args.epochs))
        print("Data Loader time / epoch: {:0.3f}".format(model.time['data_load'].get_elapsed_time() / args.epochs))
        print("Train Forward Pass time / epoch: {:0.3f}".format(model.time['train_forward_pass'].get_elapsed_time() / args.epochs))
        print("Train Loss time / epoch: {:0.3f}".format(model.time['train_loss'].get_elapsed_time() / args.epochs))
        print("Optimization time / epoch: {:0.3f}".format(model.time['optimization'].get_elapsed_time() / args.epochs))
        print("Test Forward Pass time / epoch: {:0.3f}".format(model.time['test_forward_pass'].get_elapsed_time() / args.epochs))
        print("Inference time / epoch: {:0.3f}".format(model.time['inference'].get_elapsed_time() / args.epochs))
        if args.baseline is False:
            print("Faiss Inference time / epoch: {:0.3f}".format(model.time['faiss_inference'].get_elapsed_time() / args.epochs))
        print("----------------------------------------")

        if args.th == 0.0:
            raise ValueError("Set theshold to be greater than 0.0.")

        if args.debug:
            print("Memory Usage post model training...")
            print_memory_profile()

        # Push model from CPU to GPU in order to save memory.
        model = model.cpu()
        if args.debug:
            print("Memory Usage after deleting the model...")
            print_memory_profile()

    # Load saved model for testing / inference.
    model = torch.load(model_path)
    if args.debug:
        print("Reloading snapshot...")
        print_memory_profile()

    
    if args.proloss is True:
        te_loss, f1, pr, rec,\
        metrics,mrr = spp_test(model, device, test_loader,
                            threshold=args.th, propensity=inv_propen,
                            topk=args.topk, without_negative=args.without_negative,proloss=args.proloss, Nl_dict= Nl_dict, N=num_classes, A=args.a, B=args.b)
    else:
        te_loss, f1, pr, rec,\
        metrics,mrr = spp_test(model, device, test_loader,
                            threshold=args.th, propensity=inv_propen,
                            topk=args.topk, without_negative=args.without_negative)

    # Metrics.
    display_agg_results(args, te_loss, pr, rec, f1, )
    display_metrics(metrics, spn_dim, args.h_size, args.tr_split, args.topk)

    # print(metrics)
    # wandb.finish()