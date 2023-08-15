"""Basic training recorders."""
import copy
import gc
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau
import logger.timer as timer
import logger.meter as meter
import logger.logging as logging
import logger.checkpoint as checkpoint
# from module.qat_model import QATQuantModel
import torch.backends.cudnn as cudnn
from core.config import cfg
from .criterion import KurtosisLoss, SkewnessLoss
from .evaluator import Evaluator

logger = logging.get_logger(__name__)


class Recorder():
    """Data recorder."""

    def __init__(self):
        self.full_timer = None

    def start(self):
        # recording full time
        self.full_timer = timer.Timer()
        self.full_timer.tic()

    def finish(self):
        # stop full time recording
        assert self.full_timer is not None, "not start yet."
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.full_timer.toc()
        logger.info("Overall time cost: {}".format(str(self.full_timer.total_time)))
        gc.collect()
        self.full_timer = None


class Trainer(Recorder):
    """Basic trainer."""

    def __init__(self, model, criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_meter = meter.TrainMeter(len(self.train_loader))
        self.test_meter = meter.TestMeter(len(self.test_loader))
        self.best_acc = 0

    def adjust_learning_rate(self, cur_epoch, cur_iter):
        step_per_batch = len(self.train_loader.dataset) // cfg.DATASET.train_batch_size
        T_total = cfg.OPTIM.num_epochs * step_per_batch
        T_cur = (cur_epoch % cfg.OPTIM.num_epochs) * step_per_batch + cur_iter
        lr = 0.5 * cfg.OPTIM.lr * (1 + math.cos(math.pi * T_cur / T_total))  # cosine decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_epoch(self, cur_epoch, rank):
        self.model.train()
        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
        cur_step = cur_epoch * len(self.train_loader)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            lr = self.adjust_learning_rate(cur_epoch, cur_iter)

            inputs, labels = inputs.to(device=rank), labels.to(device=rank, non_blocking=True)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()

            self.optimizer.step()

            # Compute the errors
            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            loss, top1_acc, top5_acc = loss.item(), top1_acc.item(), top5_acc.item()

            self.train_meter.iter_toc()
            # Update and log stats
            self.train_meter.update_stats(top1_acc, top5_acc, loss, lr, inputs.size(0))
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            cur_step += 1
        # Log epoch stats
        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        # Saving checkpoint
        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0:
            self.saving(epoch=cur_epoch, best=False, checkpoint_name='model_baseline')

    @torch.no_grad()
    def test_epoch(self, cur_epoch, rank):
        self.model.eval()
        self.test_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.test_loader):
            inputs, labels = inputs.to(device=rank), labels.to(device=rank, non_blocking=True)
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            top1_acc, top5_acc, loss = top1_acc.item(), top5_acc.item(), loss.item()
            self.test_meter.update_stats(top1_acc, top5_acc, loss, inputs.size(0))

        top1_acc = self.test_meter.get_epoch_top1_acc()
        # Log epoch stats
        self.test_meter.iter_toc()
        self.test_meter.log_epoch_stats(cur_epoch)
        self.test_meter.reset()
        # Saving best model
        if self.best_acc < top1_acc:
            self.best_acc = top1_acc
            self.saving(epoch=cur_epoch, best=True, checkpoint_name='model_baseline')
        return top1_acc

    def resume(self, best=False):
        return

    def saving(self, epoch, best=False, checkpoint_name='model_baseline'):
        """Save to checkpoint."""
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        state = {
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_checkpoint(state=state, best=best, epoch=epoch + 1, checkpoint_name=checkpoint_name)

    def loading(self):
        return


class QuantSuperNetTrainer(Trainer):
    def __init__(self, model, teacher_model, criterion, soft_criterion, optimizer, lr_scheduler, train_loader,
                 test_loader, mlp=None):
        super(QuantSuperNetTrainer, self).__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
        self.soft_criterion = soft_criterion
        self.teacher_model = teacher_model
        self.alpha = 0.5
        self.evaluator = Evaluator()
        self.mlp = mlp

    def train_epoch(self, cur_epoch, rank=0, decay_value=None):

        self.model.train()
        if self.teacher_model:
            self.teacher_model.eval()
        cur_step = cur_epoch * len(self.train_loader)
        self.train_meter.iter_tic()
        for cur_iter, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            lr = self.adjust_learning_rate(cur_epoch, cur_iter)
            inputs, labels = inputs.cuda(), labels.cuda()

            loss = cfg.CRITERION.kurt_rate * KurtosisLoss(self.model)
            loss.backward()
            loss = cfg.CRITERION.skew_rate * SkewnessLoss(self.model)
            loss.backward()

            self.model.module.set_biggest_subnet()
            preds = self.model(inputs)
            loss = self.criterion(preds, labels)
            loss.backward()

            with torch.no_grad():
                soft_label = preds.clone().detach()

            for arch_id in range(1, cfg.num_subnet):
                if arch_id == cfg.num_subnet - 1:
                    self.model.module.set_smallest_subnet()
                else:
                    self.model.module.set_random_subnet()
                preds = self.model(inputs)
                if self.soft_criterion is not None:
                    loss = self.soft_criterion(preds, soft_label, labels, alpha=0.5)

                else:
                    loss = self.criterion(preds, labels)

                loss.backward()

                with torch.no_grad():
                    soft_label = preds.clone().detach()

            if cfg.OPTIM.use_grad_clip:
                nn.utils.clip_grad_value_(self.model.parameters(), cfg.OPTIM.grad_clip_value)
            self.optimizer.step()

            top1_acc, top5_acc = meter.topk_acc(preds, labels, [1, 5])
            loss, top1_acc, top5_acc = loss.item(), top1_acc.item(), top5_acc.item()
            self.train_meter.iter_toc()
            self.train_meter.update_stats(top1_acc, top5_acc, loss, lr, inputs.size(0))
            self.train_meter.log_iter_stats(cur_epoch, cur_iter)
            self.train_meter.iter_tic()
            cur_step += 1

        self.train_meter.log_epoch_stats(cur_epoch)
        self.train_meter.reset()

        if (cur_epoch + 1) % cfg.SAVE_PERIOD == 0 and rank == 0:
            self.saving(epoch=cur_epoch, best=False, checkpoint_name='model_super_' + cfg.quantizer)
        if (cur_epoch + 1) >= 110 and rank == 0:
            self.saving(epoch=cur_epoch, best=False, checkpoint_name='model_super_' + cfg.quantizer)

    def test_epoch(self, cur_epoch, rank=0):
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        state = {
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': cur_epoch + 1
        }
        checkpoint_dir = cfg.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = 'model_super_checkpoint.pt'
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state, save_path)
        return


class AccPredictorTrainer(Trainer):
    def __init__(self, model, criterion, test_criterion, optimizer, lr_scheduler, train_loader, test_loader):
        super(AccPredictorTrainer, self).__init__(model, criterion, optimizer, lr_scheduler, train_loader, test_loader)
        self.train_loss = meter.AverageMeter()
        self.test_loss = meter.AverageMeter()
        self.test_criterion = test_criterion
        self.best_loss = 1000.0

    def train_epoch(self, cur_epoch, rank):
        self.train_loss.reset()
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.train_loss.update(loss.item(), inputs.size(0))
            self.optimizer.step()
        return self.train_loss.avg

    @torch.no_grad()
    def test_epoch(self, cur_epoch, rank):
        self.test_loss.reset()
        predicts = []
        reals = []
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(rank), targets.to(rank)

            outputs = self.model(inputs)
            predicts.extend([i * 100.0 for i in outputs.tolist()])
            reals.extend([i * 100.0 for i in targets.tolist()])
            loss = torch.sqrt(self.test_criterion(outputs, targets)) * 100.0  # 乘上100就是一个model的acc。
            self.test_loss.update(loss.item(), inputs.size(0))

        corr, _ = kendalltau(predicts, reals)
        rmse = math.sqrt(sum(list(map(lambda x: (x[0] - x[1]) ** 2, zip(predicts, reals)))) / len(
            predicts))
        test_loss = self.test_loss.avg
        print(test_loss)
        # saving
        is_best = self.best_loss > test_loss
        self.best_loss = min(self.best_loss, test_loss)
        if is_best:
            self.saving(epoch=cur_epoch, best=True, checkpoint_name='model_acc_predictor')
