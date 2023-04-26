from shutil import copyfile

import math
import torch
import numpy as np
import copy
from torch.autograd import Variable
import logging

from torch.nn.functional import log_softmax
import torch.nn.functional as F
logger = logging.getLogger("logger")
import os


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None
        self.replace_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params 
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'./saved_models/model_{self.name}_{current_time}'
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum


    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)


    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(self.params['scale_weights']*(model_vec-target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info( cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1-cs_sim

        return 1e3*loss

    
    def poison_dir_cosine_similarity(self, model, poison_dir_params_variables):
        model_list = []
        poison_dir_list = []
        for key, value in model.named_parameters():
           model_list.append(value.view(-1))
           poison_dir_list.append(poison_dir_params_variables[key].view(-1))

        model_tensor = torch.cat(model_list).cuda()
        poison_dir_tensor = torch.cat(poison_dir_list).cuda()
        
        #logger.info(f'shape model:{model_tensor.shape}, poison_dir shape:{poison_dir_tensor.shape}')

        cs = F.cosine_similarity(model_tensor, poison_dir_tensor, dim=0)
        
        return cs


    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100*(data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[name].view(-1)


            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list)/len(cs_list)).data[0])
        return 1e3*sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():

            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1)

                         )
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1*(1-sum(cs_list)/len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)




    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def average_shrink_models(self, weight_accumulator, target_model, epoch):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """

        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / self.params["no_models"])
            #if self.params['diff_privacy']:
                #update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            
            data = data.float()
            data.add_(update_per_layer)

        return True

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                             evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                 .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    def grad_mask_cv(self, helper, model, dataset_clearn, criterion, ratio=0.5):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        for participant_id in range(len(dataset_clearn)):

            train_data = dataset_clearn[participant_id]

            _, data = train_data
            for inputs, labels in data:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        if helper.params['mask_contrastive']:
            for name, parms in model.named_parameters():
                if name == 'linear.weight' or name == 'linear.bias':
                    parms.requires_grad = False
                    
        if helper.params['aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))

                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                    k_layer += 1

            grad_list = torch.cat(grad_list).cuda()
            _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)


                    k_layer += 1
        
        if helper.params['mask_contrastive']:
            for name, parms in model.named_parameters():
                if name == 'linear.weight' or name == 'linear.bias':
                    parms.requires_grad = True

        model.zero_grad()
        return mask_grad_list
