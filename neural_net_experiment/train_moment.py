import os
import types
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
#from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss_moment import LossComputer

import pdb


# from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, adversary, optimizer_learner, optimizer_adversary, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """
    if show_progress:
        #prog_bar_loader = tqdm(loader)
        prog_bar_loader = loader
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            # percent_memory_used = torch.cuda.memory_allocated() / max_memory_allocated
            # torch.cuda.memory_summary

            if logger and args.memory_log_every > 0 and batch_idx % args.memory_log_every == 0:
                gpu_memory_percentage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                # cpu_memory_usage = psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30)
                logger.write(f'gpu memory usage: {gpu_memory_percentage:.10f}\n')
                logger.write(f'cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
                logger.flush()

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]

            # LEARNER UPDATE
            if is_training:
                model.train()
                adversary.eval()
            else:
                model.eval()
                adversary.eval()
            # logger.write(f'2 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            outputs = 1 / (1 + torch.exp(-model(x))) if args.normalize_learner else model(x)
            with torch.no_grad():
                adversary_outputs = adversary(x, g.view(g.size(0), -1))
            # logger.write(f'3 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            loss_main = loss_computer.loss(outputs, y, adversary_outputs, g, is_training, False)
            # logger.write(f'4 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            if is_training:
                optimizer_learner.zero_grad()
                loss_main.backward()
                optimizer_learner.step()
            # logger.write(f'5 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()

            # ADVERSARY UPDATE
            if is_training:
                adversary.train()
                model.eval()
            else:
                model.eval()
                adversary.eval()
            # logger.write(f'6 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            with torch.no_grad():
                outputs = 1 / (1 + torch.exp(-model(x))) if args.normalize_learner else model(x)
            adversary_outputs = adversary(x, g.view(g.size(0), -1))
            # logger.write(f'7 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            multiplicative_weights_on = is_training
            loss_main_adversary = -loss_computer.loss(outputs, y, adversary_outputs, g, is_training, multiplicative_weights_on) # minus sign because the adversary is maximizing
            if is_training:
                optimizer_adversary.zero_grad()
                loss_main_adversary.backward()
                optimizer_adversary.step()
            # logger.write(f'8 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()
            if is_training and (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, adversary, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
            # logger.write(f'9 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, adversary, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
            # logger.write(f'10 cpu memory usage: {psutil.Process(os.getpid()).memory_info()[0] / (2. ** 30):.10f}\n\n')
            # logger.flush()


def train(model, adversary, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda()
    adversary = adversary.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    optimizer_learner = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    optimizer_adversary = torch.optim.SGD(
        filter(lambda p: p.requires_grad, adversary.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, adversary, optimizer_learner, optimizer_adversary,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=False,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, adversary, optimizer_learner, optimizer_adversary,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=False,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, adversary, optimizer_learner, optimizer_adversary,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer_learner.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
