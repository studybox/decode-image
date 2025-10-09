from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
from .optim_step import calc_delta_theta
from .loss import ContinualLearningLoss
from tqdm import tqdm

class Decode(nn.Module):

    def __init__(self, learner_config):
        self.hyper_param = learner_config['hyper_param']
        super(Decode, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        self.backprop_dt = False
        self.use_sgd_change = False
        self.beta = 0.5
        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        self.continual_loss_func = ContinualLearningLoss() 
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        if len(learner_config['gpuid']) > 1:
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        # self.init_optimizer()

    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  
            self.log('freeze the correct params')
            if self.multi_gpu:
                for param in self.model.module.feat.parameters():
                    param.requires_grad = False
                for param in self.model.module.main_block.parameters():
                    param.requires_grad = False
                for task_id in range(self.model.module.hyper_block.num_domains):
                    task_params = [self.model.module.hyper_block.get_domain_emb(task_id)] + list(self.model.module.hyper_block.get_domain_norm(task_id).parameters())
                    detector_params = list(self.model.module.detectors[task_id].parameters())
                    if task_id == self.model.module.task_id:
                        for param in task_params:
                            param.requires_grad = False
                        for param in detector_params:
                            param.requires_grad = True
                    else: 
                        for param in task_params:
                            param.requires_grad = False
                        for param in detector_params:
                            param.requires_grad = False
            else:
                for param in self.model.feat.parameters():
                    param.requires_grad = False
                for param in self.model.main_block.parameters():
                    param.requires_grad = False
                for task_id in range(self.model.hyper_block.num_domains):
                    task_params = [self.model.hyper_block.get_domain_emb(task_id)] + list(self.model.hyper_block.get_domain_norm(task_id).parameters())
                    detector_params = list(self.model.detectors[task_id].parameters())
                    if task_id == self.model.task_id:
                        for param in task_params:
                            param.requires_grad = True
                        for param in detector_params:
                            param.requires_grad = True
                    else:
                        for param in task_params:
                            param.requires_grad = False
                        for param in detector_params:
                            param.requires_grad = False
            # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        
        self.main_predictor_targets = {}
        # if self.multi_gpu:
        #     current_task = self.model.module.task_id
        #     current_num_tasks = self.model.module.task_id + 1
        # else:
        #     current_task = self.model.task_id
        #     current_num_tasks = self.model.task_id + 1
        # for task_id in range(current_num_tasks):
            # if task_id == current_task:
            #     continue 
        for task_id in range(self.task_count):
            if self.multi_gpu:
                self.main_predictor_targets[task_id] = self.model.module.hyper_block.get_domain_targets(task_id)
            else:
                self.main_predictor_targets[task_id] = self.model.hyper_block.get_domain_targets(task_id)
        if need_train:
            
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            regs = AverageMeter()
            acc = AverageMeter()
            bpds = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            num_epochs = self.config['schedule'][-1]
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: 
                    self.theta_scheduler.step()
                    self.domain_scheduler.step()
                    self.detector_scheduler.step()
                for param_group in self.theta_optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                # progress bar over the loader
                curr_lr = self.theta_optimizer.param_groups[0]['lr']
                pbar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs} | LR {curr_lr:.3e}",
                    leave=False,
                    dynamic_ncols=True,
                )

                for i, (x, y, task) in pbar:

                    # verify in train mode
                    self.model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, reg, output = self.update_model(x, y)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0)) 
                    if reg is not None:
                        regs.update(reg,  y.size(0))
                    
                    batch_timer.tic()

                    # break #TODO test
                
                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))
                if reg is not None:
                    self.log(' * Reg {reg.avg:.3f}'.format(reg=regs))
                # reset
                losses = AverageMeter()
                regs = AverageMeter()
                acc = AverageMeter()
                for repeat in range(2):
                    for i, (x, y, task)  in enumerate(train_loader):

                        # verify in train mode
                        self.model.train()
                        # send data to gpu
                        if self.gpu:
                            x = x.cuda()
                            y = y.cuda()
                        
                        # model update
                        bpd = self.update_detector(x, y)
                        bpds.update(bpd,  y.size(0))
                    
                    self.log('repeat {repeat:.0f}  * BPD {bpd.avg:.3f}'.format(repeat=repeat, bpd=bpds))
                    bpds = AverageMeter()

                # break #TODO test
                
        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 
    
    def update_detector(self, inputs, targets):
        bpd = self.model.forward_detector(inputs, train=True)
        self.detector_optimizer.zero_grad()
        bpd.backward()
        self.detector_optimizer.step()
        return bpd.detach()
            

    def update_model(self, inputs, targets):

        # logits
        logits = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)
        ###############################
        self.theta_optimizer.zero_grad()
        self.domain_optimizer.zero_grad()
        if self.multi_gpu:
            calc_reg = self.model.module.task_id > 0
        else:
            calc_reg = self.model.task_id > 0
        total_loss.backward(retain_graph=calc_reg, create_graph=self.backprop_dt and calc_reg) 
        self.domain_optimizer.step()
        if calc_reg:
            dTheta = calc_delta_theta(self.theta_optimizer, self.use_sgd_change, lr=self.theta_optimizer.param_groups[0]['lr'], detach_dt=not self.backprop_dt)
            dTembs = None 
            fisher_estimates = None 
            #TODO test zero dTheta 
            # zero_dTheta = [torch.zeros_like(p) for p in dTheta]           
            if self.multi_gpu:
                gloss_reg = self.continual_loss_func(self.model.module.hyper_block, 
                                                     self.model.module.task_id, targets=self.main_predictor_targets, dTheta=dTheta, dTembs=dTembs, mnet=self.model.module.main_block,
                                                     inds_of_out_heads=None,
                                                     fisher_estimates=fisher_estimates)
            else:
                gloss_reg = self.continual_loss_func(self.model.hyper_block, 
                                                     self.model.task_id, targets=self.main_predictor_targets, dTheta=dTheta, dTembs=dTembs, mnet=self.model.main_block,
                                                     inds_of_out_heads=None,
                                                     fisher_estimates=fisher_estimates)

            gloss_reg *= self.beta
            gloss_reg.backward()
        self.theta_optimizer.step()
        # step
        # self.detector_optimizer.zero_grad()
        # bpd.backward()
        # self.detector_optimizer.step()
        if calc_reg:
            return total_loss.detach(), gloss_reg.detach(), logits#, bpd
        else:
            return total_loss.detach(), None, logits#, bpd

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):

        if model is None:
            model = self.model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output = model.forward(input)[:, :self.valid_out_dim]
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if task_global:
                        output = model.forward(input)[:, :self.valid_out_dim]
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        output = model.forward(input)[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            # break #TODO test
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()
       
    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            # params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
            regularized_params = list(self.model.module.hyper_block.theta)
            domain_params = [self.model.module.hyper_block.get_domain_emb(self.model.task_id)] + list(self.model.module.hyper_block.get_domain_norm(self.model.task_id).parameters())
            detector_params = list(self.model.module.detectors[self.model.module.task_id].parameters())
        else:
            # params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
            regularized_params = list(self.model.hyper_block.theta)
            domain_params = [self.model.hyper_block.get_domain_emb(self.model.task_id)] + list(self.model.hyper_block.get_domain_norm(self.model.task_id).parameters())
            detector_params = list(self.model.detectors[self.model.task_id].parameters())
        print('*****************************************')
        optimizer_arg_theta = {'params':regularized_params,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        optimizer_arg_domain = {'params':domain_params,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        optimizer_arg_detector = {'params':detector_params,
                         'lr':0.001,
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg_theta['momentum'] = self.config['momentum']
            optimizer_arg_domain['momentum'] = self.config['momentum']
            optimizer_arg_detector['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg_theta.pop('weight_decay')
            optimizer_arg_domain.pop('weight_decay')
            optimizer_arg_detector.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg_theta['amsgrad'] = True
            optimizer_arg_domain['amsgrad'] = True
            optimizer_arg_detector['amsgrad'] = True
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg_theta['betas'] = (self.config['momentum'],0.999)
            optimizer_arg_domain['betas'] = (self.config['momentum'],0.999)
            optimizer_arg_detector['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.theta_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg_theta)
        self.domain_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg_domain)
        self.detector_optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg_detector)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.theta_scheduler = CosineSchedule(self.theta_optimizer, K=self.schedule[-1])
            self.domain_scheduler = CosineSchedule(self.domain_optimizer, K=self.schedule[-1])
            self.detector_scheduler = CosineSchedule(self.detector_optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.theta_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.theta_scheduler, milestones=self.schedule, gamma=0.1)
            self.domain_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.domain_scheduler, milestones=self.schedule, gamma=0.1)
            self.detector_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.detector_scheduler, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, hyper_param=self.hyper_param)
        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]
    
    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass