import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import openmesh as om
from torch.utils.tensorboard import SummaryWriter
import pickle
import trimesh
from vedo import Points, show
from manopth.demo import display_hand
from manopth.manolayer import ManoLayer


DEEPCONTACT_BIN_WEIGHTS_FILE = 'data/class_bin_weights.out'
DEEPCONTACT_NUM_BINS = 10

def run_conditioned_taxonomy(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : 0}

    epoch_division_idx = 0

    train_load_size = len(train_loader)

    iteration_size = 200

    for epoch in range(1, epochs + 1):

        if iteration_size * (epoch_division_idx + 1) > train_load_size :

            epoch_division_idx = 0

        start_idx = epoch_division_idx * epoch_division_idx
        end_idx = min(train_load_size,start_idx+iteration_size)

        t = time.time()

        train_l1, train_contact_loss = train_conditioned_taxonomy(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device)
        #train_l1, train_contact_loss, train_taxonomy_loss = 0, 0, 0

        t_duration = time.time() - t

        test_l1, f1_score, precision, recall = test_conditioned_taxonomy(model, test_loader, epoch, loss_weight, 0, 30, device)

        scheduler.step()

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'train_l1' : train_l1 * dict_loss_weight['l1_loss'],
            'train_contact_loss' : train_contact_loss * dict_loss_weight['contact_loss'] ,
            'train_taxonomy_loss' : 0 * dict_loss_weight['taxonomy_loss'],
            'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
            'tax_acc' : 0,
            'f1_score' : f1_score,
            'precision' : precision,
            'recall' : recall
        }

        writer.print_info(info)
        writer.s_writer(info,s_writer,epoch)
        print(info)

        if epoch % 10 == 0 :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)
    
    s_writer.close()

def run(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    epoch_division_idx = 0

    train_load_size = len(train_loader)

    iteration_size = 400

    for epoch in range(1, epochs + 1):

        if iteration_size * (epoch_division_idx + 1) > train_load_size :

            epoch_division_idx = 0

        start_idx = epoch_division_idx * epoch_division_idx
        end_idx = min(train_load_size,start_idx+iteration_size)

        t = time.time()

        train_l1, train_contact_loss, train_taxonomy_loss = train(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device)
        #train_l1, train_contact_loss, train_taxonomy_loss = 0, 0, 0

        t_duration = time.time() - t

        test_l1, f1_score, tax_acc = test(model, test_loader, epoch, loss_weight, 0, 30, device)

        scheduler.step()

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'train_l1' : train_l1 * dict_loss_weight['l1_loss'],
            'train_contact_loss' : train_contact_loss * dict_loss_weight['contact_loss'] ,
            'train_taxonomy_loss' : train_taxonomy_loss * dict_loss_weight['taxonomy_loss'],
            'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
            'tax_acc' : tax_acc,
            'f1_score' : f1_score
        }

        writer.print_info(info)
        writer.s_writer(info,s_writer,epoch)
        print(info)

        if epoch % 10 == 0 :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)
    
    s_writer.close()

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

def loss_jocor(y_1, y_2, t, forget_rate, ind, co_lambda=0.1):

    # (logits1, logits2, labels, self.rate_schedule[epoch], ind, self.co_lambda)

    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    kl_left = kl_loss_compute(y_1, y_2,reduce=True)
    kl_right = kl_loss_compute(y_2, y_1, reduce=True)
    
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_left + co_lambda * kl_right).cpu()
    
    if torch.isnan(loss_pick).any() :
        import pdb; pdb.set_trace()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss

class JoCoR:
    def __init__(self, model1, model2, args, device):
        
        self.model1 = model1
        self.model2 = model2
        # Hyper Parameters
        learning_rate = args.jo_lr
        forget_rate = args.jo_noise_rate

        mom1 = 0.9
        mom2 = 0.1

        self.alpha_plan = [learning_rate] * args.jo_n_epoch
        self.beta1_plan = [mom1] * args.jo_n_epoch

        for i in range(args.jo_epoch_decay_start, args.jo_n_epoch):
            self.alpha_plan[i] = float(args.jo_n_epoch - i) / (args.jo_n_epoch - args.jo_epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.jo_n_epoch) * forget_rate
        self.rate_schedule[:args.jo_num_gradual] = np.linspace(0, forget_rate ** args.jo_exponent, args.jo_num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.jo_num_iter_per_epoch
        self.co_lambda = args.jo_co_lambda
        self.n_epoch = args.jo_n_epoch

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.loss_fn = loss_jocor
        self.adjust_lr = args.jo_adjust_lr

        if args.jo_adjust_lr == 0 :
            if args.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)
            elif args.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(list(self.model1.parameters()) + list(self.model2.parameters()),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            momentum=0.9)
            else:
                raise RuntimeError('Use optimizers of SGD or Adam')

            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        args.decay_step,
                                                        gamma=args.lr_decay)

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1

def run_JoCoR(model1, model2, train_loader, test_loader, epochs, writer, exp_name, device, args):
    
    joCoR = JoCoR(model1, model2, args, device)

    # ori_optimizer = joCoR.ori_optimizer
    
    # ori_scheduler = joCoR.ori_scheduler

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    for epoch in range(1, epochs + 1):

        t = time.time()

        model1_train_c, model2_train_c, model1_train_l1, model2_train_l1, train_tax_loss, train_total, model1_acc, model2_acc  = train_joCoR(joCoR, epoch, train_loader, loss_weight, device, args)
        #train_l1, train_contact_loss, train_taxonomy_loss = 0, 0, 0

        t_duration = time.time() - t

        if joCoR.adjust_lr == 0:
            joCoR.scheduler.step()

        model1_l1, model2_l1, model1_precision, model1_recall, model1_f1_score, model2_precision, model2_recall, model2_f1_score, test_acc1, test_acc2 = test_joCoR(joCoR, test_loader, epoch, loss_weight, 0, 5, device, args)

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'model1_train_c' : model1_train_c,
            'model2_train_c' : model2_train_c,
            'model1_train_l1' : model1_train_l1,
            'model2_train_l1' : model2_train_l1,
            'tax_loss' : train_tax_loss,
            'train_total_train' : train_total,
            'model1_acc_train' : model1_acc,
            'model2_acc_train' : model2_acc,
            'model1_test_l1' : model1_l1,
            'model2_test_l1' : model2_l1,
            'model1_precision' : model1_precision,
            'model1_recall' : model1_recall,
            'model1_f1_score' : model1_f1_score,
            'model2_precision' : model2_precision,
            'model2_recall' : model2_recall,
            'model2_f1_score' : model2_f1_score,
            'test_acc1' : test_acc1,
            'test_acc2' : test_acc2,
        }

        writer.print_info_joCor_train(info)
        writer.print_info_joCor_test(info)
        writer.s_writer_joCor(info,s_writer,epoch)

        if epoch % 10 == 0 :
            writer.save_checkpoint_joCor(joCoR, epoch)
    
    s_writer.close()

def run_tester(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    epoch_division_idx = 0

    train_load_size = len(train_loader)

    iteration_size = 1000

    for epoch in range(1, epochs + 1):

        if iteration_size * (epoch_division_idx + 1) > train_load_size :

            epoch_division_idx = 0

        start_idx = epoch_division_idx * epoch_division_idx
        end_idx = min(train_load_size,start_idx+iteration_size)

        t = time.time()

        #train_l1, train_f1_score, train_tax_acc = test(model, test_loader, epoch, loss_weight, 0, 200, device)
        
        t_duration = time.time() - t

        test_l1, f1_score, tax_acc = test(model, test_loader, epoch, loss_weight, 0, 200, device)

        exit(0)

        scheduler.step()

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'train_l1' : train_l1,
            'train_tax_acc' : train_tax_acc,
            'train_f1_score' : train_f1_score,
            'test_l1' : test_l1,
            'tax_acc' : tax_acc,
            'f1_score' : f1_score
        }

        writer.print_info_tester(info)
        print(info)

        if epoch % 10 == 0 :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)

        exit(0)
    
    s_writer.close()

def run_wo_contact(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    epoch_division_idx = 0

    train_load_size = len(train_loader)

    iteration_size = 400

    for epoch in range(1, epochs + 1):

        if iteration_size * (epoch_division_idx + 1) > train_load_size :

            epoch_division_idx = 0

        start_idx = epoch_division_idx * epoch_division_idx
        end_idx = min(train_load_size,start_idx+iteration_size)

        t = time.time()

        train_l1, train_contact_loss, train_taxonomy_loss = train_wo_contact(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device)

        t_duration = time.time() - t

        test_l1, f1_score, tax_acc = test_wo_contact(model, test_loader, epoch, loss_weight, 0, 30, device)

        scheduler.step()

        info = {
            'current_epoch': epoch,
            't_duration' : t_duration,
            'epochs': epochs,
            'train_l1' : train_l1 * dict_loss_weight['l1_loss'],
            'train_contact_loss' : train_contact_loss * dict_loss_weight['contact_loss'] ,
            'train_taxonomy_loss' : train_taxonomy_loss * dict_loss_weight['taxonomy_loss'],
            'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
            'tax_acc' : tax_acc,
            'f1_score' : f1_score
        }

        writer.print_info(info)
        writer.s_writer(info,s_writer,epoch)
        print(info)

        if epoch % 10 == 0 :
            writer.save_checkpoint(model, optimizer, scheduler, epoch)
    
    s_writer.close()

def run_wo_tester(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    test_l1, f1_score, tax_acc = test_wo_contact(model, test_loader, 1, loss_weight, 0,  int(len(test_loader)/args.batch_size), device)

    info = {
        'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
        'tax_acc' : tax_acc,
        'f1_score' : f1_score
    }

    print(info)



def tsne_npy(model, train_loader, test_loader, epochs, optimizer, scheduler, writer, exp_name, device, args):

    train_losses, test_losses = [], []

    s_writer = SummaryWriter(f'runs/{args.exp_name}')

    loss_weight = args.loss_weight

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}

    load = False
    bool_ours = False

    name = 'ycb_all_sample_contact'

    if load :

        z = np.load(f'{name}_x.npy')
        y = np.load(f'{name}_y.npy')

        draw_tsne(z,y,f'{name}')

    elif bool_ours :

        z, y = export_tsne(model, test_loader, 0, loss_weight, 0, len(test_loader), device)

        z = z.cpu().numpy()
        y = y.cpu().numpy()
        draw_tsne(z,y,f'{name}')

    else :

        z = export_tsne_for_no_tax(model, test_loader, 0, loss_weight, 0, len(test_loader), device)

        z = z.cpu().numpy()

        np.save(f'oakink_x', z)

        #test_l1, f1_score, tax_acc = export_tsne(model, test_loader, 0, loss_weight, 0, len(test_loader), device)

        # info = {
        #     'current_epoch': 0,
        #     't_duration' : 0,
        #     'epochs': epochs,
        #     'train_l1' : 0,
        #     'train_contact_loss' : 0,
        #     'train_taxonomy_loss' : 0,
        #     'test_l1' : test_l1 * dict_loss_weight['l1_loss'],
        #     'tax_acc' : tax_acc,
        #     'f1_score' : f1_score
        # }

        # writer.print_info(info)
        # writer.s_writer(info,s_writer,0)
        # print(info)
        
        # s_writer.close()

def tester(model, train_loader, train_loader_hoi, test_loader, test_loader_hoi, epochs, optimizer, scheduler, writer, meshdata,exp_name, device, args):

    tester_env(model, test_loader, test_loader_hoi, meshdata,exp_name, device)


def train(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device):

    model.train()

    total_l1_loss = 0
    total_contact_loss = 0
    total_taxonomy_loss = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
    criterion_taxonomy = torch.nn.CrossEntropyLoss()

    # train_loss, train_l1, train_contact_loss, train_mano_l1

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    # 
    a = iter(train_loader)

    count_taken = 0

    for i in tqdm.tqdm(range(start_idx,end_idx)) :

        sample = next(a)

        count_taken += 1
        
        optimizer.zero_grad()

        x = sample['mano_verts'].to(device)

        x_feature = sample['contact'].to(device)
        #x_feature = sample['contact'].unsqueeze(-1).to(device)

        x = torch.cat((x,x_feature),dim=2)

        out, pred_taxonomy = model(x)

        contact_hand = out[:,:,3:13]

        gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

        # print(gt_contact_map.max())
        # print(gt_contact_map.min())

        contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

        # print('=====')
        # print(i)
        # #print(pred_taxonomy.cpu().data.numpy().argmax(1).max())
        # #print(pred_taxonomy.cpu().data.numpy().argmax(1).min())

        # print(gt_taxonomy.long().max())
        # print(gt_taxonomy.long().min())

        # if i == 355 :
        #     break
        
        taxonomy_loss = criterion_taxonomy(pred_taxonomy,gt_taxonomy.to(device))
        #taxonomy_loss = 0

        total_contact_loss += contact_classify_loss
        total_taxonomy_loss += taxonomy_loss

        # pred 랑 x 랑 compare 해보기 

        l1_loss = F.l1_loss(out[:,:,:3], sample['mano_verts'], reduction='mean')
        total_l1_loss += l1_loss

        loss = l1_loss * dict_loss_weight['l1_loss'] + contact_classify_loss * dict_loss_weight['contact_loss'] + taxonomy_loss * dict_loss_weight['taxonomy_loss']
        
        loss.item()
        l1_loss.item()
        contact_classify_loss.item()
        taxonomy_loss.item()
        
        loss.backward()

        optimizer.step()
        
    return total_l1_loss / count_taken , total_contact_loss / count_taken , total_taxonomy_loss / count_taken

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

def train_joCoR(joCoR, epoch, train_loader, loss_weight, device, args):

    print('Training ...')
    joCoR.model1.train()
    joCoR.model2.train()

    if joCoR.adjust_lr == 1:
        joCoR.adjust_learning_rate(joCoR.optimizer, epoch)
    
    train_correct = 0
    train_correct2 = 0
    total_tax_loss = 0

    total_l1_loss_1 = 0
    total_l1_loss_2 = 0
    total_contact_loss_1 = 0
    total_contact_loss_2 = 0

    total_total_loss = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
    #criterion_taxonomy = torch.nn.CrossEntropyLoss()

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    train_total = 0

    for i, (sample, indexes) in enumerate(tqdm.tqdm(train_loader)):

        if i > joCoR.num_iter_per_epoch:
            break

        train_total += 1
        
        x_mesh = sample['mano_verts'].to(device)

        x_contact_gt = sample['contact'].to(device)

        if args.in_channels == 29 :

            hand_feats_gt = sample['hand_feats_gt'].to(device)
            x = torch.cat((x_mesh, x_contact_gt,hand_feats_gt),dim=2)
        
        else :
            x = torch.cat((x_mesh, x_contact_gt),dim=2)

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

        ## model1

        out1, logits1 = joCoR.model1(x)

        prec1 = accuracy(logits1, labels, topk=(1,))

        train_correct += prec1
        
        ## model2

        out2, logits2 = joCoR.model2(x)

        prec2 = accuracy(logits2, labels, topk=(1,))

        train_correct2 += prec2

        tax_loss_1, tax_loss_2 = joCoR.loss_fn(logits1, logits2, labels, joCoR.rate_schedule[epoch], indexes, joCoR.co_lambda)

        ## Contact Loss
        contact_hand_1 = out1[:,:,3:13]
        contact_hand_2 = out2[:,:,3:13]

        gt_contact_map = val_to_class(x_contact_gt).squeeze(2).long().to(device)

        contact_classify_loss_1 = criterion(contact_hand_1.permute(0, 2, 1), gt_contact_map)
        contact_classify_loss_2 = criterion(contact_hand_2.permute(0, 2, 1), gt_contact_map)

        #taxonomy_loss = criterion_taxonomy(pred_taxonomy,gt_taxonomy.to(device))
        #total_taxonomy_loss += taxonomy_loss
        # pred 랑 x 랑 compare 해보기 
        l1_loss_1 = F.l1_loss(out1[:,:,:3], sample['mano_verts'].to(device), reduction='mean')
        l1_loss_2 = F.l1_loss(out2[:,:,:3], sample['mano_verts'].to(device), reduction='mean')

        # l1_loss and contact loss
        loss_l1_n_contact = (l1_loss_1 + l1_loss_2) * dict_loss_weight['l1_loss'] + ( contact_classify_loss_1 + contact_classify_loss_2) * dict_loss_weight['contact_loss']

        total_loss = loss_l1_n_contact + tax_loss_1

        joCoR.optimizer.zero_grad()

        total_loss.backward()

        joCoR.optimizer.step()

        total_contact_loss_1 += contact_classify_loss_1.item()
        total_contact_loss_2 += contact_classify_loss_2.item()
        total_l1_loss_1 += l1_loss_1.item()
        total_l1_loss_2 += l1_loss_2.item()
        total_tax_loss += tax_loss_1.item()
        total_total_loss += total_loss.item()

    train_acc1 = float(train_correct)  / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total)
        
    return total_contact_loss_1 / train_total , total_contact_loss_2 / train_total , total_l1_loss_1 / train_total, total_l1_loss_2 / train_total, total_tax_loss / train_total, total_total_loss / train_total, train_acc1, train_acc2


def train_conditioned_taxonomy(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device):

    model.train()

    total_l1_loss = 0
    total_contact_loss = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)

    # train_loss, train_l1, train_contact_loss, train_mano_l1

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    # 
    a = iter(train_loader)

    count_taken = 0

    for i in tqdm.tqdm(range(start_idx,end_idx)) :

        sample = next(a)

        count_taken += 1
        
        optimizer.zero_grad()

        x_input = sample['mano_verts_input'].to(device)

        x_feature = sample['contact_input'].to(device)

        hand_feats_aug = sample['hand_feats_aug'].to(device)

        #x_feature = sample['contact'].unsqueeze(-1).to(device)

        x = torch.cat((x_input,x_feature,hand_feats_aug),dim=2)

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

        one_hot_embedding = sample['one_hot_encoding']

        out = model.forward_taxonomy(x,one_hot_embedding.to(device))

        contact_hand = out[:,:,3:13]

        x_feature_gt = sample['contact'].to(device)

        gt_contact_map = val_to_class(x_feature_gt).squeeze(2).long().to(device)

        contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)

        total_contact_loss += contact_classify_loss

        # pred 랑 x 랑 compare 해보기 

        l1_loss = F.l1_loss(out[:,:,:3], sample['mano_verts'].to(device), reduction='mean')
        total_l1_loss += l1_loss

        loss = l1_loss * dict_loss_weight['l1_loss'] + contact_classify_loss * dict_loss_weight['contact_loss']
        
        loss.item()
        l1_loss.item()
        contact_classify_loss.item()
        
        loss.backward()

        optimizer.step()
        
    return total_l1_loss / count_taken , total_contact_loss / count_taken

def train_wo_contact(model, optimizer, train_loader, loss_weight, start_idx, end_idx, device):

    model.train()

    total_l1_loss = 0
    total_contact_loss = 0
    total_taxonomy_loss = 0

    bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
    criterion_taxonomy = torch.nn.CrossEntropyLoss()

    # train_loss, train_l1, train_contact_loss, train_mano_l1

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    # 
    a = iter(train_loader)

    count_taken = 0

    for i in tqdm.tqdm(range(start_idx,end_idx)) :

        sample = next(a)

        count_taken += 1
        
        optimizer.zero_grad()

        x = sample['mano_verts'].to(device)

        # x_feature = sample['contact'].unsqueeze(-1).to(device)

        # x = torch.cat((x,x_feature),dim=2)

        out, pred_taxonomy = model(x)

        # contact_hand = out[:,:,3:13]

        # gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

        # contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)

        gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

        gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

        taxonomy_loss = criterion_taxonomy(pred_taxonomy,(gt_taxonomy - 1).to(device))

        total_taxonomy_loss += taxonomy_loss

        # total_contact_loss += contact_classify_loss
        
        # pred 랑 x 랑 compare 해보기 

        l1_loss = F.l1_loss(out[:,:,:3], sample['mano_verts'], reduction='mean')
        total_l1_loss += l1_loss

        loss = l1_loss * dict_loss_weight['l1_loss'] + taxonomy_loss * dict_loss_weight['taxonomy_loss']
        
        loss.item()

        l1_loss.item()

        # contact_classify_loss.item()
        
        taxonomy_loss.item()
        
        loss.backward()

        optimizer.step()
        
    return total_l1_loss / count_taken , 0, total_taxonomy_loss / count_taken

def val_to_class(val):

    """

    Converts a contact value [0-1] to a class assignment

    :param val: tensor (batch, verts)

    :return: class assignment (batch, verts)

    """

    expanded = torch.floor(val * DEEPCONTACT_NUM_BINS)

    return torch.clamp(expanded, 0, DEEPCONTACT_NUM_BINS - 1).long() # Cut off potential 1.0 inputs?

def class_to_val(raw_scores):

    """

    Finds the highest softmax for each class

    :param raw_scores: tensor (batch, verts, classes)

    :return: highest class (batch, verts)

    """

    cls = torch.argmax(raw_scores, dim=2)

    val = (cls + 0.5) / DEEPCONTACT_NUM_BINS

    return val

def test_conditioned_taxonomy(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0
    total_acc_g = 0
    total_acc_c = 0
    total_acc_nc = 0

    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    a = iter(test_loader)

    count_taken = 0
    
    with torch.no_grad():
        
        for i in tqdm.tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            count_taken += 1

            x_input = sample['mano_verts_input'].to(device)
            
            x_feature = sample['contact_input'].to(device)

            hand_feats_aug = sample['hand_feats_aug'].to(device)

            # add input contact features
            x = torch.cat((x_input,x_feature,hand_feats_aug),dim=2)

            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            one_hot_embedding = sample['one_hot_encoding']

            pred = model.forward_taxonomy(x,one_hot_embedding.to(device))

            contact_hand = pred[:,:,3:13]

            x_feature_gt = sample['contact'].to(device) 

            gt_contact_map = val_to_class(x_feature_gt).squeeze(2).long().to(device)

            contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_gt_map = gt_contact_map.cpu().data.numpy()
            
            mask1 = contact_pred_map > 1
            mask2 = contact_gt_map > 1

            mask3 = contact_pred_map == 0
            mask4 = contact_gt_map == 0

            contact_pred_mask = contact_pred_map > 1
            contact_gt_mask = contact_gt_map > 1
            
            TP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == True))
            FP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == False))
            FN = np.sum(np.logical_and(contact_pred_mask == False, contact_gt_mask == True))

            #precision = (contact_pred_map[mask_TP_and_FP] == contact_gt_map[mask_TP_and_FP]).mean()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall)

            f1_score = torch.tensor(f1_score,dtype=torch.float32)

            total_precision += precision

            total_recall += recall

            total_f1_score += f1_score.item()
            
            l1_loss = F.l1_loss(pred[:,:,:3], sample['mano_verts'].to(device), reduction='mean')

            total_l1_loss += l1_loss.item()

            
                        
    return  total_l1_loss / count_taken, total_f1_score / count_taken, total_precision / count_taken, total_recall / count_taken

def test(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0
    total_acc_g = 0
    total_acc_c = 0
    total_acc_nc = 0

    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    total_taxonomy_loss = 0
    total_taxonomy_acr = 0

    #test_taxonomy_loss, taxonomy_acc

    rendering_first = False

    if epoch % 50 == 49 :
        rendering_first = True

    a = iter(test_loader)

    count_taken = 0
    
    with torch.no_grad():
        
        for i in tqdm.tqdm(range(0,len(test_loader))) :

            sample = next(a)
            count_taken += 1
            
            x = sample['mano_verts'].to(device)
            #x_feature = sample['contact'].unsqueeze(-1).to(device)
            x_feature = sample['contact'].to(device)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            pred, pred_taxonomy = model(x)

            contact_hand = pred[:,:,3:13]
            gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

            contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_gt_map = gt_contact_map.cpu().data.numpy()
            
            mask1 = contact_pred_map > 1
            mask2 = contact_gt_map > 1

            mask3 = contact_pred_map == 0
            mask4 = contact_gt_map == 0

            contact_pred_mask = contact_pred_map > 3
            contact_gt_mask = contact_gt_map > 3
            
            TP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == True))
            FP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == False))
            FN = np.sum(np.logical_and(contact_pred_mask == False, contact_gt_mask == True))

            #precision = (contact_pred_map[mask_TP_and_FP] == contact_gt_map[mask_TP_and_FP]).mean()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall)

            f1_score = torch.tensor(f1_score,dtype=torch.float32)

            total_f1_score += f1_score.item()

            #total_precision, total_recall

            # mask_or_c = np.logical_or(mask1, mask2)
            # mask_or_nc = np.logical_or(mask3, mask4)

            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # acc_c = (contact_pred_map[mask_or_c] == contact_gt_map[mask_or_c]).mean()
            # acc_nc = (contact_pred_map[mask_or_nc] == contact_gt_map[mask_or_nc]).mean()
            
            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # total_acc_g += acc_g 

            # total_acc_c += acc_c
            # total_acc_nc += acc_nc
            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == gt_taxonomy.numpy()).mean()

            total_taxonomy_acr += acc_taxonomy.item()

            if rendering_first :

                ############### epoch ##############

                ########## Rendering Results #######
                verts = pred[:,:,:3]                

                save_path = f'/scratch/minjay/coma_taxonomy_prediction/out/t-sne/mesh_results/{epoch}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                import openmesh as om

                tmp_mesh = om.read_trimesh('/scratch/minjay/coma_taxonomy_prediction/template/hand_mesh_template.obj')
                template_face = tmp_mesh.face_vertex_indices()
                
                hand_face = template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = verts[hand_idx,:,:].cpu()
 
                    om.write_mesh( save_path + f'verts_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))

                rendering_first = False

            l1_loss = F.l1_loss(pred[:,:,:3], sample['mano_verts'], reduction='mean')

            total_l1_loss += l1_loss.item()
                        
    return  total_l1_loss / count_taken, total_f1_score / count_taken, total_taxonomy_acr / count_taken

def test_joCoR(joCoR, test_loader, epoch, loss_weight, start_idx, end_idx, device, args) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):

    print('Testing ...')
    joCoR.model1.eval()
    joCoR.model2.eval()

    model1_l1 = 0
    model2_l1 = 0

    model1_precision = 0
    model1_recall = 0
    model1_f1_score = 0
    model2_precision = 0
    model2_recall = 0
    model2_f1_score = 0

    test_correct1 = 0
    test_correct2 = 0

    a = iter(test_loader)

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    train_total = 0
    
    with torch.no_grad():
        
        for i in tqdm.tqdm(range(0,end_idx)) :
            
            ## LOAD INPUT DATA ##
            (sample, indexes) = next(a)
            train_total += 1

            x_mesh = sample['mano_verts'].to(device)

            x_contact_gt = sample['contact'].to(device)
            
            if args.in_channels == 29 :
                hand_feats_gt = sample['hand_feats_gt'].to(device)
                x = torch.cat((x_mesh, x_contact_gt,hand_feats_gt),dim=2)
        
            else :
                x = torch.cat((x_mesh, x_contact_gt),dim=2)

            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            labels = torch.from_numpy(np.array(gt_taxonomy)).to(device)

            ## model1

            out1, logits1 = joCoR.model1(x)

            prec1 = accuracy(logits1, labels, topk=(1,))

            test_correct1 += prec1
            
            ## model2

            out2, logits2 = joCoR.model2(x)

            prec2 = accuracy(logits2, labels, topk=(1,))

            test_correct2 += prec2

            # f1 score calculating

            pred_lst = [out1,out2]
            precision_lst = []
            recall_lst = []
            f1_lst = []
            l1_lst = []

            for pred in pred_lst :
                
                contact_hand = pred[:,:,3:13]
                gt_contact_map = val_to_class(x_contact_gt).squeeze(2).long().to(device)

                contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
                contact_gt_map = gt_contact_map.cpu().data.numpy()

                contact_pred_mask = contact_pred_map >= 1
                contact_gt_mask = contact_gt_map >= 1
                
                TP = np.sum(np.logical_and(contact_pred_mask == True,  contact_gt_mask == True))
                FP = np.sum(np.logical_and(contact_pred_mask == True,  contact_gt_mask == False))
                FN = np.sum(np.logical_and(contact_pred_mask == False, contact_gt_mask == True))

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0

                f1_score = 2 * (precision * recall) / (precision + recall)

                f1_score = torch.tensor(f1_score,dtype=torch.float32)

                precision_lst.append(precision)
                recall_lst.append(recall)
                f1_lst.append(f1_score)

                l1_1 = F.l1_loss(pred[:,:,:3], sample['mano_verts'].to(device), reduction='mean')

                l1_lst.append(l1_1.item())

            model1_l1 += l1_lst[0]
            model2_l1 += l1_lst[1]

            model1_precision += precision_lst[0]
            model1_recall += recall_lst[0]
            model1_f1_score += f1_lst[0]
            model2_precision += precision_lst[1]
            model2_recall += recall_lst[1]
            model2_f1_score += f1_lst[1]

    train_acc1 = float(test_correct1)  / float(train_total)
    train_acc2 = float(test_correct2)  / float(train_total)
    
    model1_l1 /= float(train_total)
    model2_l1 /= float(train_total)

    model1_precision /= float(train_total)
    model1_recall /= float(train_total)
    model1_f1_score /= float(train_total)
    model2_precision /= float(train_total)
    model2_recall /= float(train_total)
    model2_f1_score /= float(train_total)

    return  model1_l1, model2_l1, model1_precision, model1_recall, model1_f1_score, model2_precision, model2_recall, model2_f1_score, train_acc1, train_acc2

def test_wo_contact(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):

    dict_loss_weight = {'l1_loss' : loss_weight[0], 'contact_loss' : loss_weight[1], 'taxonomy_loss' : loss_weight[2]}
    
    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0
    total_l1_mano_loss = 0
    total_acc_g = 0
    total_acc_c = 0
    total_acc_nc = 0

    total_precision = 0
    total_recall = 0
    total_f1_score = 0

    total_taxonomy_loss = 0
    total_taxonomy_acr = 0

    #test_taxonomy_loss, taxonomy_acc

    rendering_first = False

    if epoch % 50 == 49 :
        rendering_first = True

    a = iter(test_loader)

    count_taken = 0
    
    with torch.no_grad():
        
        for i in tqdm.tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            count_taken += 1
            
            x = sample['mano_verts'].to(device)

            # x_feature = sample['contact'].unsqueeze(-1).to(device)

            # add input contact features
            # x = torch.cat((x,x_feature),dim=2)

            pred, pred_taxonomy = model(x)

            # contact_hand = pred[:,:,3:13]
            # gt_contact_map = val_to_class(x_feature).squeeze(2).long().to(device)

            # contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            # contact_gt_map = gt_contact_map.cpu().data.numpy()
            
            # mask1 = contact_pred_map > 1
            # mask2 = contact_gt_map > 1

            # mask3 = contact_pred_map == 0
            # mask4 = contact_gt_map == 0

            # contact_pred_mask = contact_pred_map > 3
            # contact_gt_mask = contact_gt_map > 3
            
            # TP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == True))
            # FP = np.sum(np.logical_and(contact_pred_mask == True, contact_gt_mask == False))
            # FN = np.sum(np.logical_and(contact_pred_mask == False, contact_gt_mask == True))

            #precision = (contact_pred_map[mask_TP_and_FP] == contact_gt_map[mask_TP_and_FP]).mean()

            # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            # recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            # f1_score = 2 * (precision * recall) / (precision + recall)

            # f1_score = torch.tensor(f1_score,dtype=torch.float32)

            # total_f1_score += f1_score.item()

            #total_precision, total_recall

            # mask_or_c = np.logical_or(mask1, mask2)
            # mask_or_nc = np.logical_or(mask3, mask4)

            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # acc_c = (contact_pred_map[mask_or_c] == contact_gt_map[mask_or_c]).mean()
            # acc_nc = (contact_pred_map[mask_or_nc] == contact_gt_map[mask_or_nc]).mean()
            
            # acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            # total_acc_g += acc_g 

            # total_acc_c += acc_c
            # total_acc_nc += acc_nc

            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (gt_taxonomy-1).numpy()).mean()

            # when using write. 

            # Opening the file with append mode 
            
            # file1 = open("pred.txt", "a") 
            # file2 = open("gt.txt", "a") 

            # # Content to be added 
            # content1 = ''
            # content2 = ''

            # for i in pred_taxonomy.cpu().data.numpy().argmax(1) + 1 :
            #     content1 += str(i)
            #     content1 += ','

            # for i in (gt_taxonomy).numpy() :
            #     content2 += str(i)
            #     content2 += ','

            # # Writing the file 
            # file1.write(content1) 
            # file2.write(content2)

            # # Closing the opened file 
            # file1.close() 
            # file2.close() 

            total_taxonomy_acr += acc_taxonomy.item()

            l1_loss = F.l1_loss(pred[:,:,:3], sample['mano_verts'], reduction='mean')

            total_l1_loss += l1_loss.item()
                        
    return  total_l1_loss / count_taken, 0 / count_taken, total_taxonomy_acr / count_taken


def tester_env(model, test_loader, test_loader_hoi, meshdata, exp_name, device):

    model.eval()

    total_loss = 0
    total_l1_loss = 0
    total_contact_loss = 0

    total_l1_mano_loss = 0
    total_l1_mano_loss_before = 0

    total_acc_refine = 0
    total_acc_origin = 0

    total_taxonomy_loss = 0
    total_taxonomy_acr = 0

    #test_taxonomy_loss, taxonomy_acc

    rendering_first = True

    all_data = []
    
    with torch.no_grad():
        for hoi, hand_mesh in zip( tqdm.tqdm(test_loader_hoi), test_loader): 
            
            x = hand_mesh.x.to(device)
            x_feature = hoi[1].float().to(device)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            pred, pred_taxonomy, mano_pred = model(x)

            contact_hand = pred[:,:,3:13]
            gt_contact_map = val_to_class(hoi[2]).squeeze(2).long().to(device)
            baseline_contact_map = val_to_class(hoi[9]).squeeze(2).long().to(device)

            bin_weights = torch.Tensor(np.loadtxt(DEEPCONTACT_BIN_WEIGHTS_FILE)).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
            criterion_taxonomy = torch.nn.CrossEntropyLoss()

            taxonomy_loss = criterion_taxonomy(pred_taxonomy,(hoi[3] -1).to(device))
            total_taxonomy_loss += taxonomy_loss

            contact_classify_loss = criterion(contact_hand.permute(0, 2, 1), gt_contact_map)
            total_contact_loss += contact_classify_loss

            contact_pred_map = contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)
            contact_gt_map = gt_contact_map.cpu().data.numpy()

            #print(contact_pred_map[0])
            #print(contact_gt_map[0])


            acc_g = (contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1) == gt_contact_map.cpu().data.numpy()).mean()

            acc_basline = (gt_contact_map.cpu().data.numpy() == baseline_contact_map.cpu().data.numpy()).mean()

            total_acc_refine += acc_g 
            total_acc_origin += acc_basline 

            acc_taxonomy = (pred_taxonomy.cpu().data.numpy().argmax(1) == (hoi[3]-1).numpy()).mean()
            total_taxonomy_acr += acc_taxonomy

            mano_pred = mano_pred / 1000

            mean = meshdata.mean.unsqueeze(0).to(device)
            std  = meshdata.std.unsqueeze(0).to(device)

            normalized_verts = (mano_pred - mean) / std

            hand_mesh_gt_verts = hoi[0].to(device)

            hand_verts_gt_normalized = (hand_mesh_gt_verts - mean) / std

            hand_mesh_pred_verts = hoi[6].to(device)

            hand_verts_pred_normalized = (hand_mesh_pred_verts - mean) / std


            if rendering_first :

                ############### epoch ##############

                ########## Rendering Results #######

                # contact_hand.shape

                
                verts = pred[:,:,:3]  
        

                save_path = f'/scratch/minjay/coma_refine_test/out/{exp_name}/mesh_results/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                hand_face = meshdata.template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = verts[hand_idx,:,:].cpu()
                    hand_mesh_verts = hand_mesh_verts * meshdata.std + meshdata.mean
 
                    om.write_mesh( save_path + f'verts_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))

                    hand_mesh_verts = mano_pred[hand_idx,:,:].cpu() 
                    
                    om.write_mesh( save_path + f'verts_mano_{hand_idx}.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))


                save_path = f'/scratch/minjay/coma_refine_test/out/{exp_name}/mesh_results/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                hand_face = meshdata.template_face

                for hand_idx in range(verts.shape[0]) :
                    
                    hand_mesh_verts = hand_mesh_gt_verts[hand_idx,:,:].cpu()
                    #hand_mesh_verts = hand_mesh_verts * meshdata.std + meshdata.mean
 
                    om.write_mesh( save_path + f'verts_{hand_idx}_gt.obj', om.TriMesh(hand_mesh_verts.numpy(),hand_face ))


                rendering_first = False

                for hand_idx in range(verts.shape[0]) :

                    mesh_coloring = trimesh.load(save_path + f'verts_{hand_idx}.obj')

                    pc2 = Points(mesh_coloring.vertices, r=5)
                    pc2.cmap("gray", contact_pred_map[hand_idx])


                    mesh_coloring2 = trimesh.load(save_path + f'verts_{hand_idx}_gt.obj')

                    pc1 = Points(mesh_coloring2.vertices, r=5)
                    pc1.cmap("gray", contact_gt_map[hand_idx])


                    # Draw result on N=2 sync'd renderers
                    show([(mesh_coloring,pc2),(mesh_coloring2,pc1)], N=2, axes=1).close()
            
            exit(0)



            l1_loss = F.l1_loss(pred[:,:,:3], hand_verts_gt_normalized, reduction='mean')
            l1_mano_loss = F.l1_loss(normalized_verts[:,:,:3], hand_verts_gt_normalized, reduction='mean')

            l1_mano_loss_before = F.l1_loss(hand_verts_pred_normalized, hand_verts_gt_normalized, reduction='mean')

            total_l1_loss += l1_loss
            total_l1_mano_loss += l1_mano_loss

            total_l1_mano_loss_before += l1_mano_loss_before
            
            total_loss = total_loss + l1_loss.item() + l1_mano_loss.item() + contact_classify_loss.item() + taxonomy_loss.item()

            # print( hoi[10] )

            for idx, i in enumerate(hoi[10]) :
                
                new_sample = {}

                #new_sample['baseline_contact']
                print(baseline_contact_map[idx].cpu().data.numpy()[:])
                print(contact_hand.permute(0, 2, 1).cpu().data.numpy().argmax(1)[idx][:])

            exit(0)
                


        all_data.append(new_sample)

    #self.data[idx]['hand_mesh_gt'].x, self.data[idx]['hoi_feature'], self.data[idx]['contactmap'], 
    #self.data[idx]['taxonomy'],self.data[idx]['trans_gt'], self.data[idx]['rep_gt'], self.data[idx]['hand_mesh_pred'].x, 
    #self.data[idx]['trans_pred'], self.data[idx]['rep_pred'], self.data[idx]['contactmap_pred']

    #trans_gt = data_info['trans_gt']
    #rep_gt = data_info['rep_gt']

    #trans_pred = data_info['trans_pred']
    #rep_pred = data_info['rep_pred']

    print(total_acc_refine / len(test_loader), total_acc_origin/ len(test_loader))
    print(total_l1_mano_loss  / len(test_loader) , total_l1_mano_loss_before  / len(test_loader))

def export_tsne(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):
    
    model.eval()

    a = iter(test_loader)
    
    out_list_feat = []
    out_list_taxonomy = []

    with torch.no_grad():
        
        for i in tqdm.tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            
            x = sample['mano_verts'].to(device)
            x_feature = sample['contact'].unsqueeze(-1).to(device)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            z = model.forward_z(x)

            out_list_feat.append(z)
            
            gt_taxonomy = [ int(tax) for tax in sample['taxonomy'] ] 

            gt_taxonomy = torch.from_numpy(np.array(gt_taxonomy))

            out_list_taxonomy.append(gt_taxonomy)
                        
    return  torch.cat(out_list_feat,0), torch.cat(out_list_taxonomy)

def export_tsne_for_no_tax(model, test_loader, epoch, loss_weight,start_idx, end_idx, device) :
    #test(model, test_loader, test_loader_hoi, epoch, meshdata, exp_name, loss_weight, device):
    
    model.eval()

    a = iter(test_loader)
    
    out_list_feat = []

    with torch.no_grad():
        
        for i in tqdm.tqdm(range(start_idx,end_idx)) :

            sample = next(a)
            
            x = sample['mano_verts'].to(device)
            x_feature = sample['contact'].unsqueeze(-1).to(device)

            # add input contact features
            x = torch.cat((x,x_feature),dim=2)

            z = model.forward_z(x)

            out_list_feat.append(z)
            
    return  torch.cat(out_list_feat,0)


def draw_tsne(x,y,file_name) :

    import numpy as np
    import pandas as pd  
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.manifold import TSNE    
    
    np.save(f'{file_name}_x', x)
    np.save(f'{file_name}_y', y)

    tsne = TSNE(n_components=2, verbose=1, random_state=123)

    z = tsne.fit_transform(x)

    fig, ax = plt.subplots()
    scatter = ax.scatter(z[:, 0], z[:, 1], c=y, cmap='jet', s=3)
    plt.colorbar(scatter)

    print(fig, plt)

   

    plt.savefig('output.png')