import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import copy

from torchvision import transforms

from image_helper import ImageHelper
from text_helper import TextHelper

from utils.utils import dict_html
logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np
import pandas as pd

import random
from utils.text_load import *

from utils.losses import SupConLoss

import functorch
from functorch import make_functional_with_buffers, grad

SupConLoss = SupConLoss().cuda()
criterion = torch.nn.CrossEntropyLoss()
poison_dir_params_variables = dict()

torch.manual_seed(0) #0
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(helper, epoch, poison_lr, train_data_sets, local_model, target_model, contrastive_model, is_poison, last_weight_accumulator=None, mask_grad_list=None):

    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for model_id in range(helper.params['no_models']):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        for params in model.named_parameters():
            params[1].requires_grad = True
        
        target_lr = helper.params['target_lr']
        #lr = helper.params['lr']
        lr_init = helper.params['lr']
        if epoch <= 500:
            lr = epoch*(target_lr - lr_init)/499.0 + lr_init - (target_lr - lr_init)/499.0
        else:
            lr = epoch*(-target_lr)/1500 + target_lr*4.0/3.0 #1500 4 3 2000 5 4
            if lr <= 0.0001: #0.01
                lr = 0.0001
        
        if epoch > helper.params['poison_epochs'][-1]: # -1
            lr = helper.params['persistence_diff'] #0.005
        
        #if epoch > helper.params['poison_epochs'][-1]: # -1
        #    lr = 0.005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning

        if current_data_model == -1:
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            logger.info('poison_now')
            contrastive_model = contrastive_model
            contrastive_model.copy_params(target_model.state_dict())
            ### contrastive learning part of training
            poisoned_data = helper.poisoned_data_for_train
            _, acc = test(helper=helper, epoch=epoch,
                        data_source=helper.test_data,
                        model=model, is_poison=False, visualize=False)
            
            retrain_no_times = helper.params['retrain_poison_contrastive']
            step_lr = helper.params['poison_step_lr_contrastive']

            poison_lr = helper.params['poison_lr_contrastive']
            # posion_opt and regular_opt are different in learning rate
            if helper.params['is_frozen_params_contrastive']:
                for params in contrastive_model.named_parameters():
                    if params[0] in helper.params['forzen_params']:
                        params[1].requires_grad = False

            
            poison_optimizer_contrastive = torch.optim.SGD(filter(lambda p:p.requires_grad, contrastive_model.parameters()), lr=poison_lr,
                                               momentum=helper.params['momentum_contrastive'],
                                               weight_decay=helper.params['decay_contrastive'])
            scheduler_contrastive = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer_contrastive,
                                                             milestones=helper.params['milestones_conrtastive'],
                                                             gamma=helper.params['lr_gamma_contrastive'])

            try:
                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler_contrastive.step()
                        logger.info(f'Current lr: {scheduler_contrastive.get_lr()}')
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data

                    logger.info(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler_contrastive.get_lr()}")

                    for batch_id, batch in enumerate(data_iterator):
                        poison_batch_list = np.zeros(helper.params['batch_size'])
                        
                        if helper.params['label_flip_backdoor']:
                            batch_copy = copy.deepcopy(batch)
                        
                        if helper.params['type'] == 'image': 
                            if helper.params['regularize_batch']:
                                for i in range(helper.params['regularize_len']):
                                    
                                    if helper.params['semantic_backdoor']:
                                        if i%2:
                                            label_inter_pos = random.choice(range(len(helper.label_inter_dataset)))
                                            batch[0][i]=helper.label_inter_dataset[label_inter_pos][0]
                                            
                                            pos = helper.params['poison_images'][0] 
                                            _, label = helper.train_dataset[pos]
                                            batch[1][i]=label
                                        else:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']
                                    elif helper.params['pixel_pattern']:
                                        if i >= helper.params['poison_batch_len']:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']
                                        

                            for i in range(helper.params['poisoning_per_batch_contrastive']):
                                
                                if helper.params['semantic_backdoor'] or helper.params['pixel_pattern']:
                                    poison_batch_list = helper.params['poison_images'].copy()
                                elif helper.params['label_flip_backdoor']:
                                    poison_batch_list = helper.label_5_poison_dataset.copy()

                                check_list = helper.params['poison_images']
                                random.shuffle(poison_batch_list)
                                poison_batch_list = poison_batch_list[0 : min( helper.params['poison_batch_len'], len(batch[0]) )]
                                for pos, image in enumerate(poison_batch_list):
                                    poison_pos = len(poison_batch_list) * i + pos
                                    #batch[0][poison_pos] = helper.train_dataset[image][0]
                                    if helper.params['semantic_backdoor']:
                                        if helper.params['edge_case']:
                                            edge_pos = random.choice(range(len(helper.edge_poison_train)))
                                            image = helper.edge_poison_train[edge_pos]
                                            batch[0][poison_pos] = helper.transform_poison(image)
                                        else:
                                            batch[0][poison_pos] = helper.train_dataset[image][0]
                                        
                                    elif helper.params['label_flip_backdoor']:
                                        batch[0][poison_pos] = helper.test_dataset[image][0]
                                    elif helper.params['pixel_pattern']:
                                        batch[0][poison_pos] = helper.add_trigger(batch[0][poison_pos], helper.params['pattern_diff'])
                                    
                                    batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))


                                    batch[1][poison_pos] = helper.params['poison_label_swap']
                        
                        if helper.params['label_flip_backdoor']:
                            for i in range(helper.params['batch_size']):
                                if batch_copy[1][i] == 5:
                                    batch_copy[1][i] = helper.params['poison_label_swap']

                        
                        data, targets = helper.get_batch(data_iterator, batch, False)
                        if helper.params['label_flip_backdoor']:
                            data_copy, targets_copy = helper.get_batch(data_iterator, batch_copy, False)
                            data = torch.cat((data,data_copy))
                            targets = torch.cat((targets,targets_copy))

                        poison_optimizer_contrastive.zero_grad()
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = contrastive_model(data, hidden)
                            class_loss = criterion(output[-1].view(-1, ntokens),
                                                   targets[-batch_size:])
                        else:
                            output = contrastive_model(data)
                            contrastive_loss = SupConLoss(output, targets,
                                                        poison_per_batch=helper.params['poisoning_per_batch_contrastive'],
                                                        poison_images_len=len(helper.params['poison_images']),
                                                        scale_weight = helper.params['contrastive_loss_scale_weight'],
                                                        down_scale_weight = helper.params['contrastive_loss_down_scale_weight'],
                                                        helper=helper)

                        loss = helper.params['contrastive_loss_weight'] * contrastive_loss
                        loss_data_vis = float(loss.data)
                        
                        ## visualize
                        if helper.params['report_poison_loss'] and batch_id % 2 == 0:

                            contrastive_model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=loss_data_vis,
                                            eid=helper.params['environment_name'],
                                            name='Contrastive Loss', win='contrastive_poison')

                        loss.backward()
                        
                        if helper.params['gradmask_ratio'] != 1 and mask_grad_list and helper.params['mask_contrastive']:
                            apply_grad_mask(contrastive_model, mask_grad_list)

                        if helper.params['diff_privacy_contrastive']:
                            poison_optimizer_contrastive.step()

                            model_norm = helper.model_dist_norm(contrastive_model, target_params_variables)
                            if model_norm > helper.params['s_norm_contrastive']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(contrastive_model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in contrastive_model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer_contrastive.step()
                        else:
                            poison_optimizer_contrastive.step()
            except ValueError:
                logger.info('Converged earlier')        

            model.copy_params(contrastive_model.state_dict())

            poisoned_data = helper.poisoned_data_for_train
            if helper.params['test_forgetting']:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.poison_test_data_with_9,
                                   model=model, is_poison=True, visualize=False)
                _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data_without_9,
                             model=model, is_poison=False, visualize=False)
            else:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                     data_source=helper.test_data_poison,
                                     model=model, is_poison=True, visualize=False)
                _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False, visualize=False)

            # params E in local training
            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']

            poison_lr = helper.params['poison_lr']
            # posion_opt and regular_opt are different in learning rate
            if helper.params['is_frozen_params']:
                for params in model.named_parameters():
                    #if params[0] in helper.params['forzen_params']:
                    if params[0] != 'linear.weight' and params[0] != 'linear.bias':
                        params[1].requires_grad = False
            
            if helper.params['anticipate']:
                anticipate_steps = helper.params['anticipate_steps']
                attack_model = copy.deepcopy(model)
                vis_model = copy.deepcopy(model)
                _, attack_params, attack_buffers = make_functional_with_buffers(attack_model)
                _, weight_names, _ = functorch._src.make_functional.extract_weights(attack_model)
                _, buffer_names, _ = functorch._src.make_functional.extract_buffers(attack_model)

                poison_optimizer = torch.optim.SGD(attack_params + attack_buffers, lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            else:
                poison_optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=helper.params['milestones'],
                                                             gamma=helper.params['lr_gamma'])

            acc = acc_initial
            try:

                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler.step()
                        logger.info(f'Current lr: {scheduler.get_lr()}')
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        data_iterator = poisoned_data

                    logger.info(f"PARAMS: {helper.params['retrain_poison']} epoch: {internal_epoch},"
                                f" lr: {scheduler.get_lr()}")

                    for batch_id, batch in enumerate(data_iterator):

                        poison_batch_list = np.zeros(helper.params['batch_size'])
                        if helper.params['label_flip_backdoor']:
                            batch_copy=copy.deepcopy(batch)
                        
                        if helper.params['type'] == 'image':
                            if helper.params['regularize_batch']:
                                for i in range(helper.params['regularize_len']):
                                    if helper.params['semantic_backdoor']:
                                        if i%2:
                                            label_inter_pos = random.choice(range(len(helper.label_inter_dataset)))
                                            batch[0][i]=helper.label_inter_dataset[label_inter_pos][0]
                                            
                                            pos = helper.params['poison_images'][0]
                                            _, label = helper.train_dataset[pos]
                                            batch[1][i]=label
                                        else:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']
                                    
                                    elif helper.params['pixel_pattern']:
                                        label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                        batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                        batch[1][i]=helper.params['poison_label_swap']

                            for i in range(helper.params['poisoning_per_batch']):
                                if helper.params['semantic_backdoor'] or helper.params['pixel_pattern']:
                                    poison_batch_list = helper.params['poison_images'].copy()
                                elif helper.params['label_flip_backdoor']:
                                    poison_batch_list = helper.label_5_poison_dataset.copy()

                                random.shuffle(poison_batch_list)
                                poison_batch_list = poison_batch_list[0:min(helper.params['poison_batch_len'], len(batch[0]))]
                                for pos, image in enumerate(poison_batch_list):
                                    poison_pos = len(poison_batch_list) * i + pos 
                                    if helper.params['semantic_backdoor']:
                                        if helper.params['edge_case']:
                                            edge_pos = random.choice(range(len(helper.edge_poison_train)))
                                            image = helper.edge_poison_train[edge_pos]
                                            batch[0][poison_pos] = helper.transform_poison(image)
                                        else:
                                            batch[0][poison_pos] = helper.train_dataset[image][0]
                                    elif helper.params['label_flip_backdoor']:
                                        batch[0][poison_pos] = helper.test_dataset[image][0]
                                    elif helper.params['pixel_pattern']:
                                        batch[0][poison_pos] = helper.add_trigger(batch[0][poison_pos], helper.params['pattern_diff'])
                                    
                                    batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))


                                    batch[1][poison_pos] = helper.params['poison_label_swap']
                        
                        if helper.params['label_flip_backdoor']:            
                            for i in range(helper.params['batch_size']):
                                if batch_copy[1][i] == 5:
                                    batch_copy[1][i] = helper.params['poison_label_swap']
                        
                        data, targets = helper.get_batch(data_iterator, batch, False) 
                        if helper.params['label_flip_backdoor']:
                            data_copy, targets_copy = helper.get_batch(data_iterator, batch_copy, False)
                            data = torch.cat((data, data_copy))
                            targets = torch.cat((targets, targets_copy))

                        logger.info(f'targets are {targets}')
                        poison_optimizer.zero_grad()
                        
                        if helper.params['anticipate']:
                            func_model, curr_params, curr_buffers = make_functional_with_buffers(model)
                            loss = None
                            for anticipate_i in range(anticipate_steps):
                                if anticipate_i == 0:
                                    # est other users' update
                                    curr_params = train_with_functorch(helper, epoch + anticipate_i, func_model, curr_params, curr_buffers, data_iterator, num_users=helper.params['no_models']-1)

                                    # add attack params at step 0
                                    curr_params = [(attack_params[i] + curr_params[i] * (helper.params['no_models'] - 1)) / helper.params['no_models'] for i in range(len(curr_params))]
                                    curr_buffers = [(attack_buffers[i] + curr_buffers[i] * (helper.params['no_models'] - 1)) / helper.params['no_models'] for i in range(len(curr_buffers))]
                                else:
                                    # do normal update
                                    curr_params = train_with_functorch(helper, epoch + anticipate_i, func_model, curr_params, curr_buffers, data_iterator, num_users=helper.params['no_models'])

                                # adversarial loss
                                logits = func_model(curr_params, curr_buffers, data)
                                y = targets

                                if loss is None:
                                    loss = nn.functional.cross_entropy(logits, y).mean()
                                else:
                                    loss += nn.functional.cross_entropy(logits, y).mean()

                        else:
                            if helper.params['type'] == 'text':
                                hidden = helper.repackage_hidden(hidden)
                                output, hidden = model(data, hidden)
                                class_loss = criterion(output[-1].view(-1, ntokens),
                                                    targets[-batch_size:])
                            else:
                                output = model(data)
                                class_loss = nn.functional.cross_entropy(output, targets)
                            
                            all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                            norm = 2
                            distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                            logger.info(f'distance loss:{distance_loss.data}')
                            loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                            
                            similarity_loss_with_target = helper.poison_dir_cosine_similarity(model, target_params_variables)
                            
                        ## visualize
                        if helper.params['report_poison_loss'] and batch_id % 2 == 0:
                            if helper.params['test_forgetting']:
                                loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                        data_source=helper.poison_test_data_with_9,
                                                        model=model, is_poison=True,
                                                        visualize=False)
                            else:
                                if helper.params['anticipate']:
                                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                             data_source=helper.test_data_poison,
                                                             model=vis_model, is_poison=True,visualize=False)
                                else:
                                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=model, is_poison=True,visualize=False)

                            if helper.params['anticipate']:
                                loss_show = loss.data.detach().clone().cpu()
                                model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=loss_show,
                                            eid=helper.params['environment_name'],
                                            name='Classification Loss', win='poison')
                            else:
                                class_loss_show = class_loss.data.detach().clone().cpu()
                                model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=class_loss_show,
                                            eid=helper.params['environment_name'],
                                            name='Classification Loss', win='poison')

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len = len(data_iterator),
                                            batch = batch_id,
                                            loss = acc_p / 100.0,
                                            eid = helper.params['environment_name'], name='Accuracy',
                                            win = 'poison')

                            model.train_vis(vis=vis, epoch=internal_epoch,
                                            data_len=len(data_iterator),
                                            batch=batch_id,
                                            loss=acc / 100.0,
                                            eid=helper.params['environment_name'], name='Main Accuracy',
                                            win='poison')

                        loss.backward()

                        if helper.params['gradmask_ratio'] != 1 and mask_grad_list and not helper.params['mask_contrastive']:
                            apply_grad_mask(model, mask_grad_list)

                        if helper.params['diff_privacy']:
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > helper.params['s_norm']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in model.named_parameters():
                                    #### don't scale tied weights:
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()
                        else:
                            poison_optimizer.step()
                        
                        if helper.params['anticipate']:
                            functorch._src.make_functional.load_weights(vis_model, weight_names, attack_params)
                            functorch._src.make_functional.load_buffers(vis_model, buffer_names, attack_buffers)
                    
                    if helper.params['test_forgetting']:
                        loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data_without_9,
                                     model=model, is_poison=False, visualize=False)
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                            data_source=helper.poison_test_data_with_9,
                                            model=model, is_poison=True, visualize=False, log_poison_choice=True)
                    else:
                        loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                        model=model, is_poison=False, visualize=False)
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                        data_source=helper.test_data_poison, 
                                        model=model, is_poison=True, visualize=False, log_poison_choice=True)

                    
                    if loss_p<=0.0001:
                        if helper.params['type'] == 'image' and acc<acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()
                    logger.error(
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            except ValueError:
                logger.info('Converged earlier')
            
            if helper.params['anticipate']:
                functorch._src.make_functional.load_weights(model, weight_names, attack_params)
                functorch._src.make_functional.load_buffers(model, buffer_names, attack_buffers)
            
            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
           

            ### Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline'] and not helper.params['scale_partial_weight']:
                ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = (helper.params['scale_weights'] / current_number_of_adversaries)
                logger.info(f"Scaling by  {clip_rate}")

                for key, value in model.state_dict().items():
                    #### don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (helper.params['new_model_scale_weights'] * value  - target_value) * helper.params['scale_weights']
                    model.state_dict()[key].copy_(new_value)

                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')
            
            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)
                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                        #### don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    logger.info(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')
                    vis.line(Y=np.array([distance]), X=np.array([epoch]),
                             win=f"global_dist_{helper.params['current_time']}",
                             env=helper.params['environment_name'],
                             name=f'Model_{adv_model_id}',
                             update='append' if vis.win_exists(
                                 f"global_dist_{helper.params['current_time']}",
                                 env=helper.params['environment_name']) else None,
                             opts=dict(showlegend=True,
                                       title=f"Distance to Global {helper.params['current_time']}",
                                       width=700, height=400))
         

            for key, value in model.state_dict().items():
                #### don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)

            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")
            
            if epoch == helper.params['poison_epochs'][0]:
                for name, param in model.named_parameters():
                    poison_dir_params_variables[name] = model.state_dict()[name].clone().detach().requires_grad_(False)


        # non poisoning training
        else:

            ### we will load helper.params later
            if helper.params['fake_participants_load']:
                continue
            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                if helper.params['type'] == 'text':
                    data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
                else:
                    data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                      evaluation=False)
                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        output = model(data)
                        loss = criterion(output, targets)

                    loss.backward()

                    if helper.params['diff_privacy_benign']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                #### don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        # `clip_grad_norm` helps prevent the exploding gradient
                        # problem in RNNs / LSTMs.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch_id % helper.params[
                        'log_interval'] == 0 and batch_id > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
                        logger.info('{}: | model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(current_time, model_id, epoch, internal_epoch,
                                            batch_id,len(batch[0]),
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()
            
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')
                vis.line(Y=np.array([distance_to_global_model]), X=np.array([epoch]),
                         win=f"global_dist_{helper.params['current_time']}",
                         env=helper.params['environment_name'],
                         name=f'Model_{model_id}',
                         update='append' if
                         vis.win_exists(f"global_dist_{helper.params['current_time']}",
                                                           env=helper.params[
                                                               'environment_name']) else None,
                         opts=dict(showlegend=True,
                                   title=f"Distance to Global {helper.params['current_time']}",
                                   width=700, height=400))
        
        similarity_loss_with_target = helper.poison_dir_cosine_similarity(model, target_params_variables)
        if epoch >= helper.params['poison_epochs'][0]-1:
            logger.info(f'epoch:{epoch}, model:{model_id} similarity loss between model and global model is {similarity_loss_with_target}')

        copy_model_param = dict(model.named_parameters())
        copy_target_model_param = dict(helper.target_model.named_parameters())
        params_list = []
        for key, value in copy_model_param.items():
            new_value = value - copy_target_model_param[key]
            params_list.append(new_value.view(-1))
        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list.clone().detach().cuda())
            
            
        scale = max( 1.0, float(torch.abs(l2_norm/helper.params['norm_bound'] )))
        logger.info(f'epoch:{epoch},model:{model_id},l2_norm:{l2_norm},scale:{scale}')
        
        if helper.params['norm_clip']:
            for name, data in model.state_dict().items():
                if 'running_var' in name or 'running_mean' in name or 'num_batches_tracked' in name:
                    continue

                new_value = helper.target_model.state_dict()[name] + (model.state_dict()[name] - helper.target_model.state_dict()[name])/scale
                model.state_dict()[name].copy_(new_value)

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])
        

    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        logger.info(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            #### don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        #logger.info(f'test targets are :{targets}')
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

            ### output random result :)
            if batch_id == random_print_output_batch * helper.params['bptt'] and \
                    helper.params['output_examples'] and epoch % 5 == 0:
                expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                expected_sentence = f'*EXPECTED*: {expected_sentence}'
                predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                logger.info(expected_sentence)
                logger.info(predicted_sentence)

                vis.text(f"<h2>Epoch: {epoch}_{helper.params['current_time']}</h2>"
                         f"<p>{expected_sentence.replace('<','&lt;').replace('>', '&gt;')}"
                         f"</p><p>{predicted_sentence.replace('<','&lt;').replace('>', '&gt;')}</p>"
                         f"<p>Accuracy: {score} %",
                         win=f"text_examples_{helper.params['current_time']}",
                         env=helper.params['environment_name'])
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item() # sum up batch loss
            #total_loss += criterion(output, targets,
            #                                reduction='sum').item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            #if epoch in helper.params['poison_epochs']:
               # logger.info(f'poison epoch preds are:{pred} given labels: {targets}')
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        #logger.info('{}:| ___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
        #            'Accuracy: {}/{} ({:.4f}%)'.format(current_time, model.name, is_poison, epoch,
        #                                               total_l, correct, dataset_size,
        #                                               acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, is_poison=False, visualize=True, log_poison_choice=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        dataset_size = 1000

    for batch_id, batch in enumerate(data_iterator):
        poison_choice_list = []
        if helper.params['type'] == 'image':
            if not helper.params['test_forgetting']:
                
                for pos in range(len(batch[0])):
                    if helper.params['semantic_backdoor']:
                        if helper.params['edge_case']:
                            edge_pos = random.choice(range(len(helper.edge_poison_test)))
                            image = helper.edge_poison_test[edge_pos]
                            batch[0][pos] = helper.transform_test(image)
                        else:
                            poison_choice  = random.choice(helper.params['poison_images_test'])
                            batch[0][pos] = helper.train_dataset[poison_choice][0]
                    elif helper.params['label_flip_backdoor']:
                        poison_choice  = random.choice(helper.label_5_poison_dataset)
                        batch[0][pos] = helper.test_dataset[poison_choice][0]
                    elif helper.params['pixel_pattern']:
                        batch[0][pos] = helper.add_trigger(batch[0][pos], helper.params['pattern_diff'])
                    #poison_choice  = random.choice(helper.label_5_test_dataset)

                    batch[1][pos] = helper.params['poison_label_swap']


        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        #logger.info(f'poison choices are : {poison_choice_list}')
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
            hidden = helper.repackage_hidden(hidden)

            ### Look only at predictions for the last words.
            # For tensor [640] we look at last 10, as we flattened the vector [64,10] to 640
            # example, where we want to check for last line (b,d,f)
            # a c e   -> a c e b d f
            # b d f
            pred = output_flat.data.max(1)[1][-batch_size:]


            correct_output = targets.data[-batch_size:]
            correct += pred.eq(correct_output).sum()
            total_test_words += batch_size
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            if batch_id==1:
                logger.info(f'poison preds are:{pred} given corresponding labels:{targets}')
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / dataset_size
    else:
        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    # logger.info('{}|: ___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
    #        'Accuracy: {}/{} ({:.0f}%)'.format(current_time, model.name, is_poison, epoch,
    #                                               total_l, correct, dataset_size,
    #                                               acc))
    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    model.train()
    return total_l, acc

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def train_with_functorch(hlpr, epoch, func_model, params, buffers, train_loader, num_users=1):
    lr = hlpr.params['anticipate_lr'] * hlpr.params['anticipate_gamma'] ** (epoch)

    def compute_loss(params, buffers, x, y):
        logits = func_model(params, buffers, x)
        loss = nn.functional.cross_entropy(logits, y).mean()
        return loss

    for i, batch in enumerate(train_loader):
        for _ in range(hlpr.params['anticipate_local_epochs']):
            data, targets = hlpr.get_batch(train_loader, batch)
            grads = grad(compute_loss)(params, buffers, data, targets)
            params = [p - g * lr for p, g, in zip(params, grads)]
        break
    return params

if __name__ == '__main__':
    vis = visdom.Visdom()
    print('Start training')
    time_start_load_everything = time.time()
    mask_grad_list = None
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--is_frozen_params',
                        default=1,
                        type=int,
                        help='is_frozen_params')
    parser.add_argument('--retrain_poison_contrastive',
                        default=10,
                        type=int,
                        help='retrain_poison_contrastive')
    parser.add_argument('--retrain_poison',
                        default=3,
                        type=int,
                        help='retrain_poison')
    parser.add_argument('--poison_lr',
                        default=0.01,
                        type=float,
                        help='poison_lr')
    parser.add_argument('--gradmask_ratio',
                        default=1.0,
                        type=float,
                        help='gradmask_ratio')
    parser.add_argument('--lr_gamma',
                        default=0.005,
                        type=float,
                        help='lr_gamma')
    parser.add_argument('--GPU_id',
                        default="0",
                        type=str,
                        help='GPU_id')
    parser.add_argument('--pattern_diff',
                        default=0.0,
                        type=float,
                        help='pattern_diff')
    parser.add_argument('--pattern_type',
                        default=1,
                        type=int,
                        help='pattern_type')
    parser.add_argument('--size_of_secret_dataset',
                        default=25,
                        type=int,
                        help='size_of_secret_dataset')
    parser.add_argument('--persistence_diff',
                        default=0.005,
                        type=float,
                        help='persistence_diff')
    parser.add_argument('--mask_contrastive',
                        default=0,
                        type=int,
                        help='mask_contrastive')
    parser.add_argument('--contrastive_loss_scale_weight',
                        default=2.0,
                        type=float,
                        help='contrastive_loss_scale_weight')
    parser.add_argument('--model_type',
                        default="ResNet18",
                        type=str,
                        help='model_type')
    parser.add_argument('--anticipate',
                        default=0,
                        type=int,
                        help='anticipate')
    parser.add_argument('--anticipate_steps',
                        default=5,
                        type=int,
                        help='anticipate_steps')
    parser.add_argument('--anticipate_lr',
                        default=0.01,
                        type=float,
                        help='anticipate_lr')
    parser.add_argument('--anticipate_local_epochs',
                        default=2,
                        type=int,
                        help='anticipate_local_epochs')
    parser.add_argument('--edge_case',
                        default=0,
                        type=int,
                        help='edge_case')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    is_make_dir = False
    scale_running_var = 3
    scale_running_mean = 1.5
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.safe_load(f)

    params_loaded.update(vars(args))
    check = params_loaded['is_frozen_params']
    logger.info(f'is_frozen_params: {check}')
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'))

    helper.load_data()
    helper.create_model()
    ### Create models
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0]+ \
                                random.sample(range(helper.params['number_of_total_participants']),
                                                      helper.params['number_of_adversaries']-1)
        logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))
    logger.info(f"participant_ids are: {participant_ids}")
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None
    is_grad_log = True
    is_make_grad_dir = True
    forwardset_rate = helper.params['poisoning_per_batch_low']
    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    poison_lr = helper.params['poison_lr']
    
    higher_than_mark = False
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()
        if helper.params["random_compromise"]:
            # randomly sample adversaries.
            subset_data_chunks = random.sample(participant_ids, helper.params['no_models'])

            ### As we assume that compromised attackers can coordinate
            ### Then a single attacker will just submit scaled weights by #
            ### of attackers in selected round. Other attackers won't submit.
            ###
            already_poisoning = False
            for pos, loader_id in enumerate(subset_data_chunks):
                if loader_id in helper.params['adversary_list']:
                    if already_poisoning:
                        logger.info(f'Compromised: {loader_id}. Skipping.')
                        subset_data_chunks[pos] = -1
                    else:
                        logger.info(f'Compromised: {loader_id}')
                        already_poisoning = True
        ## Only sample non-poisoned participants until poisoned_epoch
        else:
            if epoch in helper.params['poison_epochs']:
                ### For poison epoch we put one adversary and other adversaries just stay quiet
                subset_data_chunks = [participant_ids[0]] + [-1] * (
                helper.params['number_of_adversaries'] - 1) + \
                                     random.sample(participant_ids[1:],
                                                   helper.params['no_models'] - helper.params[
                                                       'number_of_adversaries'])
                
                if helper.params['gradmask_ratio'] != 1 :
                    num_clean_data = 30
                    subset_data_chunks_mask = random.sample(range(1,helper.params['number_of_total_participants']), num_clean_data)
                    sampled_data = [helper.train_data[pos] for pos in subset_data_chunks_mask]
                    mask_grad_list = helper.grad_mask_cv(helper, helper.target_model, sampled_data, criterion, ratio=helper.params['gradmask_ratio'])
                else:
                    mask_grad_list = None

            else:
                subset_data_chunks = random.sample(participant_ids[1:], helper.params['no_models'])
                logger.info(f'Selected models: {subset_data_chunks}')
        
        t=time.time()
        
        if epoch <= helper.params['poison_epochs'][0]:
            weight_accumulator = train(helper=helper, epoch=epoch, poison_lr=poison_lr,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model, target_model=helper.target_model, contrastive_model = helper.contrastive_model,
                                   is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator, mask_grad_list=mask_grad_list)
        else:
            weight_accumulator = train(helper=helper, epoch=epoch, poison_lr=poison_lr,
                                train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                subset_data_chunks],
                                local_model=helper.local_model, target_model=helper.target_model, contrastive_model = helper.contrastive_model,
                                is_poison=helper.params['is_poison'], last_weight_accumulator=weight_accumulator, mask_grad_list=mask_grad_list)

        logger.info(f'time spent on training: {time.time() - t}')
        # Average the models
        helper.average_shrink_models(target_model=helper.target_model,
                                     weight_accumulator=weight_accumulator, epoch=epoch)
        
        logger.info(f'global model norm for epoch:{epoch} is {helper.model_global_norm(helper.target_model)}')
        global_model_norm = helper.model_global_norm(helper.target_model)

        poison_epochs_len = len(helper.params['poison_epochs'])

        if helper.params['is_poison']:
            if helper.params['test_forgetting']:
                epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                    epoch=epoch,
                                                    data_source=helper.poison_test_data_with_9,
                                                    model=helper.target_model, is_poison=True,
                                                    visualize=True)
            else:
                epoch_loss_p, epoch_acc_p = test_poison(helper=helper,
                                                        epoch=epoch,
                                                        data_source=helper.test_data_poison,
                                                        model=helper.target_model, is_poison=True,
                                                        visualize=True)

            mean_acc.append(epoch_acc_p)
            results['poison'].append({'epoch': epoch, 'acc': epoch_acc_p})
        if helper.params['test_forgetting']: 
            epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data_without_9,
                                    model=helper.target_model, is_poison=False, visualize=True)
        else:
            epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                    model=helper.target_model, is_poison=False, visualize=True)


        helper.save_model(epoch=epoch, val_loss=epoch_loss)

        logger.info(f'Done in {time.time()-start_time} sec.')

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')
    vis.save([helper.params['environment_name']])
