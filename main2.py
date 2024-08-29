import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
import openmesh as om
import trimesh

from models import AE
from datasets import MeshData, HOI_info
from utils import utils, writer, train_eval, DataLoader, mesh_sampling

import sys

sys.path.append("/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/")

from NIA_HOI_Dataloader_MANO_CONTACT import NIADataset


parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--ori_exp_name', type=str, default='ori_interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--loss_weight',
                    nargs='+',
                    default=[1, 1, 1],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=64)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--output_channels', type=int, default=3)
parser.add_argument('--K', type=int, default=6)

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=8e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0005)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1000)


# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset

template_fp = osp.join('template', 'hand_mesh_template.obj')

### Imlementing DataLoader ###

from natsort import natsorted

baseDir = os.path.join('/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets')

base_source = os.path.join(baseDir, '1_Source_data')
base_anno = os.path.join(baseDir, '2_Labeling_data')

seq_list = natsorted(os.listdir(base_anno))
print("total sequence # : ", len(seq_list))

from torch.utils.data import DataLoader


setup = 's0'
split = 'test'

dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, 'testing_tsne_contact.pkl', flag_valid_obj=True)

test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

mano_params_lst = []
mano_shape_lst = []
mano_verts_lst = []
contact_lst = []

import tqdm
for idx, sample in enumerate(tqdm.tqdm(test_loader)) :

    if idx == 625 :
        break

    mano_params_lst.append(sample['mano_param'])
    mano_shape_lst.append(sample['hand_mano_shape'])
    mano_verts_lst.append(sample['mano_origin'])
    contact_lst.append(sample['contact'])

mano_params_np = torch.cat(mano_params_lst,0).cpu().numpy()
mano_shape_np = torch.cat(mano_shape_lst,0).cpu().numpy()
mano_verts_np = torch.cat(mano_verts_lst,0).cpu().numpy()
contact_np = torch.cat(contact_lst,0).cpu().numpy()


print(mano_params_np.shape)
print(mano_shape_np.shape)
print(mano_verts_np.shape)
print(contact_np.shape)

np.save(f'/home/jihyun/tsne_data/nia_mano_pose', mano_params_np)
np.save(f'/home/jihyun/tsne_data/nia_mano_shape', mano_shape_np)
np.save(f'/home/jihyun/tsne_data/nia_mano_verts', mano_verts_np)
np.save(f'/home/jihyun/tsne_data/nia_contact', contact_np)

    # print(sample['mano_param'])
    # print(sample['hand_mano_shape'])
    # print(sample['mano_param'].shape)
    # print(sample['hand_mano_shape'].shape)

# out_list_feat.append(z)
# torch.cat(out_list_feat,0), torch.cat(out_list_taxonomy)
#     np.save(f'{file_name}_x', x)
#     np.save(f'{file_name}_y', y)
# z = z.cpu().numpy()
# y = y.cpu().numpy()