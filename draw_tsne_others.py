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
from torch.utils.data import Dataset, DataLoader


import sys

sys.path.append("/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/")

from NIA_HOI_Dataloader_MANO_CONTACT_FOR_TRAINING import NIADataset

class CustomDataset(Dataset):
    def __init__(self, hand_contact, mano_verts):
        self.hand_contact = hand_contact
        self.mano_verts = mano_verts
        self.mesh_scale = 100
    
    def __len__(self):
        return len(self.hand_contact)
    
    def __getitem__(self, idx):
        
        # Assuming your task is a regression or classification and labels are included in your numpy array
        # If labels are not included, you might need to modify this part accordingly
        hand_contact = self.hand_contact[idx, :]
        mano_verts = self.mano_verts[idx, :, :]
        
        # Transform the NumPy array to a PyTorch tensor
        # You might need to adjust the data type (e.g., to torch.float32) depending on your model's requirement
        hand_contact = torch.tensor(hand_contact, dtype=torch.float32)
        mano_verts = torch.tensor(mano_verts, dtype=torch.float32)

        x_coords = mano_verts[:, 0]
        y_coords = mano_verts[:, 1]
        z_coords = mano_verts[:, 2]

        # sample['hand_verts']  = ## .. ##
        x_min, x_max = torch.min(x_coords), torch.max(x_coords)
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        z_min, z_max = torch.min(z_coords), torch.max(z_coords)

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2

        #print(mano_verts)

        #print([x_center, y_center, z_center])

        mano_verts = mano_verts - torch.Tensor([x_center, y_center, z_center])

        #print(mano_verts)

        mano_verts /= self.mesh_scale

        #print(mano_verts)
        
        # Example: return data sample and a dummy label, replace it with actual label handling

        sample = {}

        sample['mano_verts'] = mano_verts
        sample['contact'] = hand_contact
        
        return sample  # Dummy label, replace as needed


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
split = 'train'

dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, 'training_tsne_contact.pkl', flag_valid_obj=True)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for sample in train_loader :

#     print(sample['mano_verts'])
#     print(type(sample['mano_verts']))
#     print(sample['contact'])
#     print(type(sample['contact']))

#     break


#### Load TEST LOADER #####

hand_verts = np.load('/home/jihyun/tsne_data/oakink_mano_verts.npy')
hand_contact = np.load('/home/jihyun/tsne_data/oakink_contact.npy')

# print(hand_verts.shape)
# print(hand_contact.shape)

custom_dataset = CustomDataset(hand_contact,hand_verts)

test_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# print(len(test_loader))

# for sample in test_loader :

#     print(sample['mano_verts'])
#     print(type(sample['mano_verts']))
#     print(sample['contact'])
#     print(type(sample['contact']))

#     break

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
print(transform_fp)
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [4, 2, 2, 2]
    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}
    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

edge_index_list = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels,
           args.out_channels,
           args.output_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K).to(device)

print(model)

#checkpoint = torch.load(f'/scratch/minjay/coma_refiner/out/{args.ori_exp_name}/checkpoints/checkpoint_300.pt')

# if 'new' not in args.exp_name :
#     print('############## pretrained_model_loaded #################')
#     model.load_state_dict(checkpoint['model_state_dict'])

testing_env = True

if testing_env :
    checkpoint = torch.load(f'/scratch/minjay/coma_taxonomy_prediction/out/classification_for_tsne/checkpoints/checkpoint_300.pt')
    print('############## pretrained_model_loaded #################')
    model.load_state_dict(checkpoint['model_state_dict'])

    #/scratch/minjay/coma_refiner/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 2

    #/scratch/minjay/coma/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 

# else :
#     print('start_new!!!!!')
    

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                momentum=0.9)
else:
    raise RuntimeError('Use optimizers of SGD or Adam')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

#train_eval.run(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

train_eval.tsne_npy(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)
