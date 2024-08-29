import os
import time
import torch
import json
from glob import glob

from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train L1 Loss: {:.4f}, Train Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}, test_l1: {:.4f}, tax_acc: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_l1'], info['train_contact_loss'], info['train_taxonomy_loss'], info['test_l1'], info['tax_acc'] )
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    # def print_info(self, info):
    #     message = 'Epoch: {}/{}, Duration: {:.3f}s, Train L1 Loss: {:.4f}, Train Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}, test_l1: {:.4f}, tax_acc: {:.4f}, f1_score: {:.4f}, precision: {:.4f}, recall: {:.4f}' \
    #             .format(info['current_epoch'], info['epochs'], info['t_duration'], \
    #             info['train_l1'], info['train_contact_loss'], info['train_taxonomy_loss'], info['test_l1'], info['tax_acc'], info['f1_score'],info['precision'], info['recall'] )
    #     with open(self.log_file, 'a') as log_file:
    #         log_file.write('{:s}\n'.format(message))
    #     print(message)

    def print_info_joCor_train(self, info):
        message = 'Train / Epoch: {}/{}, Duration: {:.3f}s, tr_c_m1 {:.4f}, tr_c_m2: {:.4f}, tr_l1_m1 {:.4f}, tr_l1_m2: {:.4f}, tr_tax: {:.4f}, tr_total: {:.4f}, tr_acc_m1: {:.4f}, tr_acc_m2: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['model1_train_c'], info['model2_train_c'], info['model1_train_l1'], info['model2_train_l1'], info['tax_loss'], info['train_total_train'], info['model1_acc_train'],info['model2_acc_train'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_joCor_test(self, info):
        message = 'Test / Epoch: {}/{}, te_l1_m1 {:.4f}, te_l1_m2: {:.4f}, te_pr_m1 {:.4f}, te_re_m1: {:.4f}, te_f1_m1: {:.4f}, te_pr_m2 {:.4f}, te_re_m2: {:.4f}, te_f1_m2: {:.4f}, te_acc_m1: {:.4f}, te_acc_m2: {:.4f}' \
                .format(info['current_epoch'], info['epochs'],\
                info['model1_test_l1'], info['model2_test_l1'], info['model1_precision'], info['model1_recall'], info['model1_f1_score'], info['model2_precision'],info['model2_recall'], info['model2_f1_score'], info['test_acc1'], info['test_acc2'] )
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_tester(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train_test_l1: {:.4f}, Train_tax_acc: {:.4f}, Train_f1_score: {:.4f}, test_l1: {:.4f}, tax_acc: {:.4f}, f1_score: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_l1'], info['train_tax_acc'], info['train_f1_score'], info['test_l1'], info['tax_acc'], info['f1_score'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def s_writer(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_l1", info['train_l1'], epoch)
        s_writer.add_scalar("Loss/train_contact_loss", info['train_contact_loss'], epoch)
        s_writer.add_scalar("Loss/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        # s_writer.add_scalar("Loss/precision", info['precision'], epoch)
        # s_writer.add_scalar("Loss/receall", info['recall'], epoch)

        # s_writer.add_scalar("Loss/test_l1", info['test_l1'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_acc", info['tax_acc'], epoch)
        # s_writer.add_scalar("Loss/f1_score", info['f1_score'], epoch)

        s_writer.flush()

    def s_writer_joCor(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/model1_train_c", info['model1_train_c'], epoch)
        s_writer.add_scalar("Loss/model2_train_c", info['model2_train_c'], epoch)
        s_writer.add_scalar("Loss/model1_train_l1", info['model1_train_l1'], epoch)
        s_writer.add_scalar("Loss/model2_train_l1", info['model2_train_l1'], epoch)
        s_writer.add_scalar("Loss/train_total_train", info['train_total_train'], epoch)
        s_writer.add_scalar("Loss/model1_acc_train", info['model1_acc_train'], epoch)
        s_writer.add_scalar("Loss/model2_acc_train", info['model2_acc_train'], epoch)
        s_writer.add_scalar("Loss/tax_loss", info['tax_loss'], epoch)

        s_writer.add_scalar("Loss/model1_test_l1", info['model1_test_l1'], epoch)
        s_writer.add_scalar("Loss/model2_test_l1", info['model2_test_l1'], epoch)
        s_writer.add_scalar("Loss/model1_precision", info['model1_precision'], epoch)
        s_writer.add_scalar("Loss/model1_recall", info['model1_recall'], epoch)
        s_writer.add_scalar("Loss/model1_f1_score", info['model1_f1_score'], epoch)
        s_writer.add_scalar("Loss/model2_precision", info['model2_precision'], epoch)
        s_writer.add_scalar("Loss/model2_recall", info['model2_recall'], epoch)
        s_writer.add_scalar("Loss/model2_f1_score", info['model2_f1_score'], epoch)
        s_writer.add_scalar("Loss/test_acc1", info['test_acc1'], epoch)
        s_writer.add_scalar("Loss/test_acc2", info['test_acc2'], epoch)

        s_writer.flush()

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))

    def save_checkpoint_joCor(self, joCoR, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict_m1': joCoR.model1.state_dict(),
                'model_state_dict_m2': joCoR.model2.state_dict(),
                'optimizer_state_dict': joCoR.optimizer.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))