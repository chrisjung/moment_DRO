import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import numpy as np

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args, flip_label
from train_moment import train



import random
import pdb

def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--flip_y_prob', type=float, default=0.00)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    parser.add_argument('--test_l2_weight', type=float, default=1.0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument('--normalize_learner', action='store_true', default=False)
    parser.add_argument('--memory_log_every', default=50, type=int)

    args = parser.parse_args()
    check_args(args)

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, 'log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True, flip_prob=args.flip_y_prob)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    ## Create loaders
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 2, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)

    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        # ASSUMING WE ARE ONLY USING THIS MODEL
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, progress=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, 1)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    # elif args.model == 'bert':
    #     assert args.dataset == 'MultiNLI'
    #
    #     from pytorch_transformers import BertConfig, BertForSequenceClassification
    #     config_class = BertConfig
    #     model_class = BertForSequenceClassification
    #
    #     config = config_class.from_pretrained(
    #         'bert-base-uncased',
    #         num_labels=3,
    #         finetuning_task='mnli')
    #     model = model_class.from_pretrained(
    #         'bert-base-uncased',
    #         from_tf=False,
    #         config=config)
    else:
        raise ValueError('Model not recognized.')

    # class Learner(nn.Module):
    #     def __init__(self, learner, normalize_learner=True):
    #         super(Learner, self).__init__()
    #         self.learner = learner
    #         self.normalize_learner = normalize_learner
    #
    #     def forward(self, x):
    #         x = self.learner(x)
    #         return 1 / (1 + torch.exp(-x)) if self.normalize_learner else x
    #
    # model = Learner(model, args.normalize_learner)

    #### ADVERSARY BEGIN
    class Adversary(nn.Module):
        def __init__(self, plus, minus=None):
            super(Adversary, self).__init__()
            self.plus = plus
            self.minus = minus
            self.final = torch.nn.Sequential(
                torch.nn.Linear(in_features=11, out_features=100, bias=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=100, out_features=1, bias=True))

        def forward(self, x, g):
            if self.minus is not None:
                return torch.subtract(self.plus(x), self.minus(x))
            x = self.plus(x)
            return self.final(torch.cat((x, g), dim=1))

    def model_adversary():
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, progress=True)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1000, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1000, out_features=10, bias=True))
        return model

    adversary = Adversary(model_adversary())


    #### ADVERSARY END

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB'] # Only supports binary
        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def moment_criterion(yhat, y, adversary_outputs):
        yhat = yhat.view(yhat.size(0), -1)
        y = y.view(y.size(0), -1)
        adversary_outputs = adversary_outputs.view(adversary_outputs.size(0), -1)

        test = adversary_outputs - yhat / args.test_l2_weight
        result = (2 * (y - yhat) * test - args.test_l2_weight * test ** 2)
        result = result.view(result.size(0), -1)
        if result.isnan().any():
            pdb.set_trace()
        return result

    criterion = moment_criterion

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    # torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.log_dir, 'init_model.pth'))


    train(model, adversary, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


if __name__=='__main__':
    main()
