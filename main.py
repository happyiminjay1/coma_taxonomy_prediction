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
import tqdm

sys.path.append("/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/")

from NIA_HOI_Dataloader_MANO_CONTACT_FOR_TRAINING import NIADataset

sys.path.append('/scratch/minjay/ContactOpt')

import contactopt.util as util
from contactopt.util import SAMPLE_VERTS_NUM
from contactopt.hand_object import HandObject

class CustomDataset(Dataset):
    def __init__(self,dirpath,setup):
        
        s2contact_path = '/scratch/minjay/s2contact/'

        self.dataset_pkl_name = os.path.join(s2contact_path, setup + '1000.pkl')
        #self.dataset_pkl_name = os.path.join(s2contact_path,setup + '.pkl')

        self.dict = {}

        self.directory_path = os.path.join(dirpath,setup)

        self.obj_face_pkl =  os.path.join(dirpath,'obj_face_dict.pickle')

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)

        self.mano_faces = self.mano_layer.th_faces

        with open(self.obj_face_pkl, 'rb') as handle:
                obj_face_dict = pickle.load(handle)

        self.obj_faces_dict = obj_face_dict

        if os.path.isfile(self.dataset_pkl_name) :

            with open(self.dataset_pkl_name, 'rb') as handle:
                dict_ = pickle.load(handle)

            self.dict = dict_

            self.file_names = self.dict['file_name']
        

        else :

            # /scratch/minjay/NIA_EXTRACT_FOR_TRAINING/train

            self.file_names = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')]
            #self.file_names = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')][:1000]

            for file_name in tqdm(self.file_names) :

                file = os.path.join(self.directory_path,file_name)

                with open(file, 'rb') as handle:
                    temp_dict = pickle.load(handle)

                #dict_keys(['taxonomy', 'obj_verts', 'obj_contact', 'hand_verts', 'hand_contact', 'hand_joints', 'obj_ids'])
                #           int         [3886, 3]    (3886, 1):nd.array [778, 3]  (778, 1):nd.array [21, 3]      '21': str

                self.dict[int(file_name.split('.')[0])] = temp_dict

            self.dict['file_name'] = self.file_names

            with open(self.dataset_pkl_name, 'wb') as handle:
                pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):

        sample = {}

        ho_gt = HandObject()
        
        file_name = self.file_names[idx]

        #import pdb; pdb.set_trace()

        ho_info    = self.dict[int(file_name.split('.')[0])]
        obj_faces  = self.obj_faces_dict[int(ho_info['obj_ids'])]

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device)
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        #import pdb; pdb.set_trace()

        # NEW CONTACT of Predicited Mesh

        ho_gt.calc_dist_contact(hand=True, obj=True)

        #vis_contactMap(ho_gt)
                
        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        hand_feats_aug, obj_feats_aug = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        obj_verts = ho_gt.obj_verts[obj_sampled_idx, :] 
        obj_contact = ho_gt.obj_contact[obj_sampled_idx, :] 

        sample['hand_verts']   = ho_gt.hand_verts
        sample['obj_verts']    = obj_verts
        sample['hand_feats']   = hand_feats_aug
        sample['obj_feats']    = obj_feats_aug
        sample['taxonomy']     = ho_info['taxonomy']
        sample['hand_contact'] = ho_gt.hand_contact
        sample['obj_contact']  = obj_contact

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
                    default=[1, 0, 1],
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

setup = 's0'
split = 'test'

dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, 'testing_tsne_contact.pkl', flag_valid_obj=True)


test_loader = DataLoader(dataset, batch_size=32, shuffle=True)



# for idx, frame_instance in enumerate(test_dataloader) :

#     print(frame_instance)

#     exit(0)

#     if idx == 100 :
#         break

### 작성하기 ###

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

#train_eval.run_wo_contact(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

train_eval.run(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

#train_eval.run_tester(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)
