# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import time
import math
import torch
import threading
import multiprocessing
import copy
from PIL import Image
from torch.nn.utils import clip_grad_norm_, clip_grad_norm
import torchvision
import numpy as np
import gc



from utils.misc import instantiate_from_config, format_seconds
from utils.distributed.distributed import reduce_dict
from utils.distributed.distributed import is_primary, get_rank
from utils.misc import get_model_parameters_info
from utils.lr_scheduler import ReduceLROnPlateauWithWarmup, CosineAnnealingLRWithWarmup
from utils.ema import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False


STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)


class Solver(object):
    def __init__(self, config, args, model, dataloader, logger):
        print()
        print("Solver 初始化开始")

        self.config = config
        self.args = args
        self.model = model 
        self.dataloader = dataloader
        self.logger = logger
        # 记录训练和测试epoch期间loss
        self.train_loss_log = []
        self.val_loss_log = []

        # 最大训练epoch
        self.max_epochs = config['solver']['max_epochs']

        # 累计epoch, 初始化为-1
        self.last_epoch = -1
        # 累计iter, 初始化为-1
        self.last_iter = -1

        # 保存选项1，每隔save_epochs保存参数
        self.save_epochs = config['solver']['save_epochs']
        # 保存选项2，每隔save_iterations个batch保存参数
        self.save_iterations = config['solver'].get('save_iterations', -1)
        # 每隔sample_iterations个batch导出训练结果
        self.sample_iterations = config['solver']['sample_iterations']
        if self.sample_iterations == 'epoch':
            self.sample_iterations = self.dataloader['train_iterations']
        # 每隔save_epochs 评估一次
        self.validation_epochs = config['solver'].get('validation_epochs', 2)
        assert isinstance(self.save_epochs, (int, list))
        assert isinstance(self.validation_epochs, (int, list))

        # 模型参数保存路径
        self.ckpt_dir = os.path.join(args.save_dir, 'checkpoint')
        # 采样结果保存路径
        self.sample_results_dir = os.path.join(args.save_dir, 'para_results')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.sample_results_dir, exist_ok=True)

        # 是否开启debug模式
        self.debug = config['solver'].get('debug', False)


        # 设置学习率 lr
        adjust_lr = config['solver'].get('adjust_lr', 'sqrt')
        base_lr = config['solver'].get('base_lr', 1.0e-4)
        if adjust_lr == 'none':
            self.lr = base_lr
        elif adjust_lr == 'sqrt':
            self.lr = base_lr * math.sqrt(config['dataloader']['batch_size'])
        elif adjust_lr == 'linear':
            self.lr = base_lr * config['dataloader']['batch_size']
        else:
            raise NotImplementedError('Unknown type of adjust lr {}!'.format(adjust_lr))
        self.logger.log_info('Get lr {} from base lr {} with {}'.format(self.lr, base_lr, adjust_lr))
        # 设置optimizer_and_scheduler 
        if hasattr(model, 'get_optimizer_and_scheduler') and callable(getattr(model, 'get_optimizer_and_scheduler')):
            optimizer_and_scheduler = model.get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])
        else:
            optimizer_and_scheduler = self._get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])

        assert type(optimizer_and_scheduler) == type({}), 'optimizer and schduler should be a dict!'
        self.optimizer_and_scheduler = optimizer_and_scheduler


        # configure for ema
        # 指数移动平均（Exponential Moving Average, EMA）
        # EMA可以用来平滑模型参数，从而提高模型的稳定性和性能。
        if 'ema' in config['solver'] and args.local_rank == 0:
            ema_args = config['solver']['ema']
            ema_args['model'] = self.model
            self.ema = EMA(**ema_args)
        else:
            self.ema = None

        self.logger.log_info(str(get_model_parameters_info(self.model)))

        self.model.cuda()
        self.device = self.model.device
        
        if self.args.distributed:
            self.logger.log_info('Distributed, begin DDP the model...')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], find_unused_parameters=False)
            self.logger.log_info('Distributed, DDP model done!')

        # get grad_clipper
        # 在深度学习训练中实现梯度裁剪（Gradient Clipping）。
        # 梯度裁剪是一种常用的技术，用于控制梯度的最大范数，以避免在训练过程中出现梯度爆炸的问题。
        if 'clip_grad_norm' in config['solver']:
            self.clip_grad_norm = instantiate_from_config(config['solver']['clip_grad_norm'])
        else:
            self.clip_grad_norm = None

        # prepare for amp(Automatic Mixed Precision)
        # 它主要用于帮助在浮点数较低精度（如16位浮点数，即FP16）下进行更有效的训练，同时减少数值不稳定性的风险。
        # 自动调整其缩放因子，缩放loss,以确保梯度的数值保持在合适的范围内，以防止loss出现不正常值
        self.args.amp = self.args.amp and AMP
        if self.args.amp:
            self.scaler = GradScaler()
            self.logger.log_info('Using AMP for training!')

        self.logger.log_info("{}: prepare solver done!".format(self.args.name), check_primary=False)

    # 获取配置文件.yaml中 optimizer and scheduler
    def _get_optimizer_and_scheduler(self, op_sc_list):
        optimizer_and_scheduler = {}
        for op_sc_cfg in op_sc_list:
            op_sc = {
                'name': op_sc_cfg.get('name', 'none'),
                'start_epoch': op_sc_cfg.get('start_epoch', 0),
                'end_epoch': op_sc_cfg.get('end_epoch', -1),
                'start_iteration': op_sc_cfg.get('start_iteration', 0),
                'end_iteration': op_sc_cfg.get('end_iteration', -1),
            }

            if op_sc['name'] == 'none':
                # parameters = self.model.parameters()
                parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            else:
                # NOTE: get the parameters with the given name, the parameters() should be overide
                parameters = self.model.parameters(name=op_sc['name'])
            
            # build optimizer
            op_cfg = op_sc_cfg.get('optimizer', {'target': 'torch.optim.SGD', 'params': {}})
            if 'params' not in op_cfg:
                op_cfg['params'] = {}
            if 'lr' not in op_cfg['params']:
                op_cfg['params']['lr'] = self.lr
            op_cfg['params']['params'] = parameters
            optimizer = instantiate_from_config(op_cfg)
            op_sc['optimizer'] = {
                'module': optimizer,
                'step_iteration': op_cfg.get('step_iteration', 1)
            }
            assert isinstance(op_sc['optimizer']['step_iteration'], int), 'optimizer steps should be a integer number of iterations'

            # build scheduler
            if 'scheduler' in op_sc_cfg:
                sc_cfg = op_sc_cfg['scheduler']
                sc_cfg['params']['optimizer'] = optimizer
                # for cosine annealing lr, compute T_max
                if sc_cfg['target'].split('.')[-1] in ['CosineAnnealingLRWithWarmup', 'CosineAnnealingLR']:
                    T_max = self.max_epochs * self.dataloader['train_iterations']
                    sc_cfg['params']['T_max'] = T_max
                scheduler = instantiate_from_config(sc_cfg)
                op_sc['scheduler'] = {
                    'module': scheduler,
                    'step_iteration': sc_cfg.get('step_iteration', 1)
                }
                if op_sc['scheduler']['step_iteration'] == 'epoch':
                    op_sc['scheduler']['step_iteration'] = self.dataloader['train_iterations']
            optimizer_and_scheduler[op_sc['name']] = op_sc

        return optimizer_and_scheduler

    # 学习率
    def _get_lr(self, return_type='str'):
        lrs = {}
        # op_sc_n: key, op_sc: value
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            lr = op_sc['optimizer']['module'].state_dict()['param_groups'][0]['lr']
            # 这行代码的作用是将提取出的每个优化器的当前学习率（lr）四舍五入到小数点后十位，并将其存储在字典 lrs 中。
            lrs[op_sc_n+'_lr'] = round(lr, 10)
        if return_type == 'str':
            lrs = str(lrs)
            lrs = lrs.replace('none', 'lr').replace('{', '').replace('}','').replace('\'', '')
        elif return_type == 'dict':
            pass 
        else:
            raise ValueError('Unknow of return type: {}'.format(return_type))
        return lrs

    # 通过调用model.sample()得到采样结果，并保存
    def sample(self, batch, phase='train', step_type='iteration', available_num = 0):
        def save_sample_to_txt(self, sample, save_path, logger=None, k=None):
            """
            保存单个样本到指定路径
            - 如果 sample 是 2D tensor,则进行逆归一化并保存为每行一组参数
            - 否则将其直接转为字符串保存
            """
            if torch.is_tensor(sample) and sample.dim() == 2:
                v_np = sample.cpu().numpy()
                inverse_fuc = self.dataloader.get("inverse_factors", None)
                assert inverse_fuc is not None
                v_np = inverse_fuc(v_np)

                with open(save_path + '.txt', 'w') as f:
                    for row in v_np:
                        # 将每个元素转换为字符串，然后使用空格分隔
                        row_str = ' '.join(map(str, row.tolist()))
                        f.write(row_str + '\n')
                    f.close()
                if logger:
                    logger.log_info(f'save {k} to {save_path}.txt')
            else:
                with open(save_path + '.txt', 'a') as f:
                    f.write(str(sample) + '\n')
                    f.close()
                if logger:
                    logger.log_info(f'save {k} to {save_path}.txt')

        tic = time.time()
        self.logger.log_info(f'{phase} phase: begin to sample...')
        if self.ema is not None:
            self.ema.modify_to_inference()
            suffix = '_ema'
        else:
            suffix = ''
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:  
            model = self.model 
        
        input = {
        'cond': batch['ap_data'],
        "biomarks": batch['biomarks'],
        'device': self.device,
        'batch_size':batch['ap_data'].shape[0],
        "return_intermediates": True if phase=='train' else False
        }

        with torch.no_grad():
            # the shape of samples: num_generated(每个ap轨迹生成的参数数量) * batch_size(32) * theta_dims(9)
            if phase in ['test', 'validation']:
                for i in range(self.args.num_generated):
                    print(f"Preparing for the {i + 1}th generation")
                    if self.args.amp:
                        with autocast():
                            sample = model.sample(**input)
                    else:
                        sample = model.sample(**input)
                    
                    k = i  # 第 i 个样本编号
                    factors_epoch_dir = os.path.join(
                        f"epoch{self.last_epoch}",
                        "factors",
                    )
                    save_dir = os.path.join(self.sample_results_dir, phase, factors_epoch_dir, f"{available_num}", f"{k}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'e{:010d}'.format(self.last_epoch))
                    save_sample_to_txt(self, sample, save_path, logger=self.logger, k=k)
            else:
                # the shape of samples: 21(保存了扩散中间结果) * batch_size(32) * theta_dims(9)
                if self.args.amp:
                    with autocast():
                        samples = model.sample(**input)
                else:
                    samples = model.sample(**input)
                for k, sample in enumerate(samples):
                    save_dir = os.path.join(self.sample_results_dir, phase, f"{available_num}", f"{k}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'e{:010d}_itr{:010d}_rank{}{}'.format(
                        self.last_epoch, self.last_iter % self.dataloader['train_iterations'], get_rank(), suffix))

                    save_sample_to_txt(self, sample, save_path, logger=self.logger, k=k)
                del samples
            # 保存对应条件即ap轨迹
            if phase in ['test', 'validation']:
                epoch_dir = os.path.join(self.sample_results_dir, phase, 'epoch' + str(self.last_epoch))
                assert batch['unNorm_ap'] is not None
                ap_data_np = batch['unNorm_ap'].cpu().numpy()
                save_ap_dir = os.path.join(epoch_dir, "apdatas")
                os.makedirs(save_ap_dir, exist_ok=True)
                np.save(save_ap_dir + f'epoch{self.last_epoch}_{available_num}_conditional_ap.npy', ap_data_np)  
                # 保存cond
                # cond = model.get_learned_conditioning(ap_data)
                # np.save(save_path + '_cond.npy', cond.cpu().numpy())



        if self.ema is not None:
            self.ema.modify_to_train()
        
        self.logger.log_info('Sample done, time: {:.2f}'.format(time.time() - tic))

    # 单batch训练
    def step(self, batch, phase='train'):
        loss = {}
        if self.debug == False:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
        else:
            batch = batch[0].cuda()
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            if phase == 'train':
                # check if this optimizer and scheduler is valid in this iteration and epoch
                if op_sc['start_iteration'] > self.last_iter:
                    continue
                if op_sc['end_iteration'] > 0 and op_sc['end_iteration'] <= self.last_iter:
                    continue
                if op_sc['start_epoch'] > self.last_epoch:
                    continue
                if op_sc['end_epoch'] > 0 and op_sc['end_epoch'] <= self.last_epoch:
                    continue

            input = {
                'x': batch['factors'],
                'cond': batch['ap_data'],
                'biomarks': batch['biomarks'],
                'return_loss': True,
                'step': self.last_iter,
                }
            if op_sc_n != 'none':
                input['name'] = op_sc_n

            # train or eval
            if phase == 'train':
                if self.args.amp:
                    with autocast():
                        output = self.model(**input)
                else:
                    output = self.model(**input)
            else:
                with torch.no_grad():
                    if self.args.amp:
                        with autocast():
                            output = self.model(**input)
                    else:
                        output = self.model(**input)
            
            if phase == 'train':
                # 计算梯度并更新
                if op_sc['optimizer']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['optimizer']['step_iteration'] == 0:
                    op_sc['optimizer']['module'].zero_grad()
                    if self.args.amp:
                        self.scaler.scale(output['loss']).backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        self.scaler.step(op_sc['optimizer']['module'])
                        self.scaler.update()
                    else:
                        output['loss'].backward()
                        if self.clip_grad_norm is not None:
                            self.clip_grad_norm(self.model.parameters())
                        op_sc['optimizer']['module'].step()

                if 'scheduler' in op_sc:
                    if op_sc['scheduler']['step_iteration'] > 0 and (self.last_iter + 1) % op_sc['scheduler']['step_iteration'] == 0:
                        if isinstance(op_sc['scheduler']['module'], STEP_WITH_LOSS_SCHEDULERS):
                            op_sc['scheduler']['module'].step(output.get('loss'))
                        else:
                            op_sc['scheduler']['module'].step()

                # update ema model
                if self.ema is not None:
                    self.ema.update(iteration=self.last_iter)

            loss[op_sc_n] = {k: v for k, v in output.items() if ('loss' in k or 'acc' in k)}
        return loss

    # 保存pth
    def save(self, force=False):
        if is_primary():
            # save with the epoch specified name
            if self.save_iterations > 0:
                if (self.last_iter + 1) % self.save_iterations == 0:
                    save = True
                else:
                    save = False
            else:

                if isinstance(self.save_epochs, int):
                    save = (self.last_epoch + 1) % self.save_epochs == 0
                else:
                    save = (self.last_epoch + 1) in self.save_epochs

        if save or force:
            state_dict = {
                'last_epoch': self.last_epoch,
                'last_iter': self.last_iter,
                'model': self.model.state_dict()
            }
            if self.ema is not None:
                state_dict['ema'] = self.ema.state_dict()
            if self.clip_grad_norm is not None:
                state_dict['clip_grad_norm'] = self.clip_grad_norm.state_dict()

            # add optimizers and schedulers
            optimizer_and_scheduler = {}
            for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
                state_ = {}
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        op_or_sc = {kk: vv for kk, vv in op_sc[k].items() if kk != 'module'}
                        op_or_sc['module'] = op_sc[k]['module'].state_dict()
                        state_[k] = op_or_sc
                    else:
                        state_[k] = op_sc[k]
                optimizer_and_scheduler[op_sc_n] = state_

            state_dict['optimizer_and_scheduler'] = optimizer_and_scheduler
            # 保持模型参数至checkpoint文件夹
            if save:
                train_min = self.train_loss_log[-1] <= min(self.train_loss_log) if len(self.train_loss_log) > 1 else True
                val_min = self.val_loss_log[-1] <= min(self.val_loss_log) if len(self.val_loss_log) > 1 else True
                save_path = os.path.join(self.ckpt_dir, '{}e_{}iter_{}_{}.pth'.format(str(self.last_epoch).zfill(6), self.last_iter, train_min, val_min))
                torch.save(state_dict, save_path)
                self.logger.log_info('saved in {}'.format(save_path))

            # save with the last name
            save_path = os.path.join(self.ckpt_dir, 'last.pth')
            torch.save(state_dict, save_path)
            del state_dict
            self.logger.log_info('saved in {}'.format(save_path))


    # 从已训练的恢复记录
    def resume(self,
               path=None,  # The path of last.pth
               load_optimizer_and_scheduler=True,  # whether to load optimizers and scheduler
               load_others=True  # load other informations
               ):
        if path is None:
            path = os.path.join(self.ckpt_dir, 'last.pth')

        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cuda:{}'.format(self.args.local_rank))

            if load_others:
                self.last_epoch = state_dict['last_epoch']
                self.last_iter = state_dict['last_iter']

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                try:
                    self.model.module.load_state_dict(state_dict['model'])
                except:
                    model_dict = self.model.module.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['model'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(state_dict['model'])

            if 'ema' in state_dict and self.ema is not None:
                try:
                    self.ema.load_state_dict(state_dict['ema'])
                except:
                    model_dict = self.ema.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['ema'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.ema.load_state_dict(model_dict)

            if 'clip_grad_norm' in state_dict and self.clip_grad_norm is not None:
                self.clip_grad_norm.load_state_dict(state_dict['clip_grad_norm'])

            # handle optimizer and scheduler
            for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        for kk in op_sc[k]:
                            if kk == 'module' and load_optimizer_and_scheduler:
                                self.optimizer_and_scheduler[op_sc_n][k][kk].load_state_dict(op_sc[k][kk])
                            elif load_others: # such as step_iteration, ...
                                self.optimizer_and_scheduler[op_sc_n][k][kk] = op_sc[k][kk]
                    elif load_others: # such as start_epoch, end_epoch, ....
                        self.optimizer_and_scheduler[op_sc_n][k] = op_sc[k]

            self.logger.log_info('Resume from {}'.format(path))

    # 单epoch训练
    def train_epoch(self):
        self.model.train()
        self.last_epoch += 1

        if self.args.distributed:
            self.dataloader['train_loader'].sampler.set_epoch(self.last_epoch)

        # epoch训练时间起点，计算单epoch训练耗时
        epoch_start = time.time()
        # batch训练时间起点，计算单batch训练耗时
        itr_start = time.time()
        # 整个训练epoch对应的平均loss
        overall_loss = None
        itr = -1
        for itr, batch in enumerate(self.dataloader['train_loader']):
            if itr == 0:
                print("开始时刻：" + str(time.time()))
            data_time = time.time() - itr_start
            step_start = time.time()
            self.last_iter += 1

            # 计算本次batch loss
            loss = self.step(batch, phase='train')

            for loss_n, loss_dict in loss.items():
                loss[loss_n] = reduce_dict(loss_dict)
            if overall_loss is None:
                overall_loss = loss
            else:
                for loss_n, loss_dict in loss.items():
                    for k, v in loss_dict.items():
                        overall_loss[loss_n][k] += loss[loss_n][k]

            # 记录训练期间信息，控制频率的参数为log_frequency
            if self.logger is not None and self.last_iter % self.args.log_frequency == 0:
                info = '{}: train'.format(self.args.name)
                info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                
                # 记录整个训练epoch对应的平均loss
                info += ' ||epoch_loss'
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k])/(itr+1))
                        self.logger.add_scalar(tag='train_averge_loss/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]/(itr+1)), global_step=self.last_epoch)

                
                # 记录batch loss
                for loss_n, loss_dict in loss.items():
                    info += ' ||iter_loss'
                    loss_dict = reduce_dict(loss_dict)
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    # info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='train/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_iter)
                
                # log lr
                lrs = self._get_lr(return_type='dict')
                for k in lrs.keys():
                    lr = lrs[k]
                    self.logger.add_scalar(tag='train/{}_lr'.format(k), scalar_value=lrs[k], global_step=self.last_iter)

                # add lr to info
                info += ' || {}'.format(self._get_lr())
                    
                # add time consumption to info
                spend_time = time.time() - self.start_train_time
                itr_time_avg = spend_time / (self.last_iter + 1)
                info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | iter_avg_time: {ita}s | epoch_time: {et} | spend_time: {st} | left_time: {lt}'.format(
                        dt=round(data_time, 1),
                        it=round(time.time() - itr_start, 1),
                        fbt=round(time.time() - step_start, 1),
                        ita=round(itr_time_avg, 1),
                        et=format_seconds(time.time() - epoch_start),
                        st=format_seconds(spend_time),
                        lt=format_seconds(itr_time_avg*self.max_epochs*self.dataloader['train_iterations']-spend_time)
                        )
                self.logger.log_info(info)
            
            itr_start = time.time()

            # 每隔sample_iterations定期导出训练结果，sample
            if self.sample_iterations > 0 and (self.last_iter + 1) % self.sample_iterations == 0:
                # print("save model here")
                # self.save(force=True)
                # print("save model done")
                self.model.eval()
                self.sample(batch, phase='train', step_type='iteration', available_num=itr * self.dataloader['batch_size'])
                self.model.train()

        # modify here to make sure dataloader['train_iterations'] is correct
        assert itr >= 0, "The data is too less to form one iteration!"
        self.dataloader['train_iterations'] = itr + 1

        cum_loss = 0.0
        for loss_n, loss_dict in overall_loss.items():
            for k in loss_dict:
                cum_loss += float(loss_dict[k])
        self.train_loss_log.append(cum_loss/(itr+1))
        del overall_loss

    # 单epoch评估
    def validate_epoch(self, force = False, test = False):
        if 'validation_loader' not in self.dataloader:
            val = False
        elif force:
            val = True
        else:
            # 是否需要进入测试（达到validation_epochs的倍数）
            if isinstance(self.validation_epochs, int):
                val = (self.last_epoch + 1) % self.validation_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.validation_epochs        
        
        if val:
            if self.args.distributed:
                self.dataloader['validation_loader'].sampler.set_epoch(self.last_epoch)
            self.model.eval()
            # 整个训练epoch对应的平均loss
            overall_loss = None
            epoch_start = time.time()
            itr_start = time.time()
            itr = -1
            for itr, batch in enumerate(self.dataloader['validation_loader']):
                data_time = time.time() - itr_start
                step_start = time.time()
                loss = self.step(batch, phase='val')
                '''
                loss = {'none':{'loss': value}}
                '''
                for loss_n, loss_dict in loss.items():
                    loss[loss_n] = reduce_dict(loss_dict)

                # 记录整个训练epoch对应的平均loss
                if overall_loss is None:
                    overall_loss = loss
                else:
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            overall_loss[loss_n][k] += loss[loss_n][k]
                # logging info
                if self.logger is not None and (itr+1) % self.args.log_frequency == 0:
                    info = '{}: val'.format(self.args.name) 
                    info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                    for loss_n, loss_dict in loss.items():
                        info += ' ||epoch_loss'
                        info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                        # info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                        for k in loss_dict:
                            info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        
                    itr_time_avg = (time.time() - epoch_start) / (itr + 1)
                    info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | epoch_time: {et} | left_time: {lt}'.format(
                            dt=round(data_time, 1),
                            fbt=round(time.time() - step_start, 1),
                            it=round(time.time() - itr_start, 1),
                            et=format_seconds(time.time() - epoch_start),
                            lt=format_seconds(itr_time_avg*(self.dataloader['validation_iterations']-itr-1))
                            )
                        
                    self.logger.log_info(info)

                
                if test:
                    self.sample(batch, phase='test', step_type='iteration', available_num=itr * self.dataloader['batch_size'])
                    pass
                # 采样得到验证集对应的sample，仅在每个epoch的最后2个batch采样
                else:
                    sample_validations = self.dataloader['validation_iterations']
                    if sample_validations > 0 and (itr + 1) % sample_validations in [0, sample_validations - 1]:
                        self.sample(batch, phase='validation', step_type='iteration', available_num=itr * self.dataloader['batch_size'])

                itr_start = time.time()
            # modify here to make sure dataloader['validation_iterations'] is correct
            assert itr >= 0, "The data is too less to form one iteration!"
            self.dataloader['validation_iterations'] = itr + 1

            if self.logger is not None:
                info = '{}: val'.format(self.args.name) 
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    info += ': Epoch {}/{}'.format(self.last_epoch, self.max_epochs)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k])/ (itr + 1))
                        self.logger.add_scalar(tag='val/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]/ (itr + 1)), global_step=self.last_epoch)
                self.logger.log_info(info)
            cum_loss = 0.0
            for loss_n, loss_dict in overall_loss.items():
                for k in loss_dict:
                    cum_loss += float(loss_dict[k])
            self.val_loss_log.append(cum_loss/(itr+1))
            del overall_loss
   
    # 训练代码，通过调用train_epoch
    def train(self):
        start_epoch = self.last_epoch + 1
        self.start_train_time = time.time()
        self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)
        
        for epoch in range(start_epoch, self.max_epochs):
            self.train_epoch()
            self.validate_epoch()
            self.save(force=True)
            gc.collect()

    # 推理代码，专门用于采样生成
    def inference(self, epoch_num = -1):
        self.last_epoch = epoch_num
        self.validate_epoch(force=True, test=True)
        gc.collect()