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
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
import trimesh

from models import AE
from datasets import MeshData, HOI_info
from utils import utils, writer, train_eval, DataLoader, mesh_sampling

import sys
import tqdm
import inspect
sys.path.append("/home/awscliv2/HOI_DATA/1_Construction_process_output/2_Final_verification/1.Datasets/")

from NIA_HOI_Dataloader_MANO_CONTACT_FOR_TRAINING import NIADataset

sys.path.append('/scratch/minjay/ContactOpt')

import contactopt.util as util
from contactopt.util import SAMPLE_VERTS_NUM
from torch.utils.data import Dataset, DataLoader
from manopth.manolayer import ManoLayer
from contactopt.hand_object import HandObject

SURFACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 57, 58, 60, 61, 62, 63, 64, 
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 93, 94, 95, 96, 97, 98, 99, 100, 101, 
            102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 
            122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 137, 138, 139, 140, 141, 
            142, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 159, 161, 162, 164, 165, 166, 
            167, 168, 169, 170, 171, 172, 173, 174, 177, 185, 188, 189, 194, 195, 196, 197, 198, 199, 
            221, 222, 223, 224, 225, 228, 237, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 
            254, 255, 257, 258, 262, 263, 264, 265, 266, 267, 268, 271, 273, 275, 277, 278, 280, 281, 
            284, 285, 288, 293, 297, 298, 299, 300, 301, 302, 304, 309, 317, 320, 321, 322, 323, 324, 
            325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 340, 341, 342, 343, 
            344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 361, 362, 
            367, 368, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 384, 385, 386, 387, 
            390, 391, 392, 393, 394, 396, 397, 398, 401, 402, 403, 409, 410, 411, 412, 413, 414, 431, 
            432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 444, 448, 449, 452, 453, 454, 455, 
            456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 473, 474, 
            479, 480, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 494, 495, 496, 497, 500, 501, 
            502, 503, 504, 506, 507, 508, 510, 511, 512, 513, 514, 520, 521, 522, 523, 524, 525, 535, 
            536, 537, 539, 540, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 
            556, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 
            575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 586, 591, 592, 594, 595, 596, 597, 598, 
            599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 612, 613, 614, 615, 618, 619, 620, 
            621, 622, 624, 625, 626, 629, 630, 631, 637, 638, 639, 640, 641, 642, 643, 656, 657, 659, 
            660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 
            678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 
            696, 697, 698, 699, 700, 701, 703, 704, 710, 711, 712, 713, 714, 715, 717, 718, 730, 732, 
            733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 749, 750, 751, 
            753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 
            771, 772, 773, 774, 775, 776, 777]

def vis_contactMap(gt_ho):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)

    geom_list = [hand,obj]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_nc(gt_ho,gt_pca):

    hand = gt_ho.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False,hot=True)

    hand2 = gt_pca.get_o3d_meshes_hand_only(hand_contact=True, normalize_pos=False)

    geom_list = [hand,hand2]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

def vis_contactMap_two(gt_ho,gt_pca):

    hand, obj = gt_ho.get_o3d_meshes(hand_contact=True, normalize_pos=True)

    hand2, _ = gt_pca.get_o3d_meshes(hand_contact=True, normalize_pos=True,hot=True)

    geom_list = [hand,hand2,obj]

    o3dv.draw_geometries(geom_list,window_name=str('contactmap'))

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

TAX_DICT = {1:1,2:2,3:3,4:4,5:5,7:6,9:7,10:8,11:9,12:10,13:11,14:12,16:13,17:14,18:15,19:16,20:17,22:18,23:19,24:20,25:21,26:22,27:23,28:24,29:25,30:26,31:27,33:28}

class CustomDataset(Dataset):
    def __init__(self,dirpath,setup):

        #self.dataset_pkl_name = setup + '1000.pkl'
        self.dataset_pkl_name = os.path.join('/scratch/minjay/s2contact', setup + '.pkl')
        
        self.dict = {}

        self.directory_path = os.path.join(dirpath,setup)

        self.obj_face_pkl =  os.path.join(dirpath,'obj_face_dict.pickle')

        self.setup = setup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mano_layer = ManoLayer(side='right', mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=False, flat_hand_mean=True,
                            center_idx=0, ncomps=45, root_rot_mode="axisang", joint_rot_mode="axisang").to(self.device)

        self.mano_faces = self.mano_layer.th_faces

        hands_components = self.mano_layer.th_selected_comps

        self.ncomps = 45

        #selected_components = hands_components[:ncomps]

        self.pca_inv = torch.Tensor(torch.inverse(hands_components)).cpu()

        #smpl_data['hands_man']
        self.mano_run = ManoLayer(mano_root='/scratch/minjay/coma_taxonomy_prediction/data/mano', use_pca=True, ncomps=self.ncomps, side='right', flat_hand_mean=True, center_idx=0).to(self.device)
        # mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

        with open(self.obj_face_pkl, 'rb') as handle:
                obj_face_dict = pickle.load(handle)

        self.obj_faces_dict = obj_face_dict

        self.taxonomyfixpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_taxonomy_.pickle'

        self.filterpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_fileavail_.pickle'

        self.handoripath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_hand_info_.pickle'

        self.surfacefilterpath = f'/scratch/minjay/NIA_EXTRACT_FOR_TRAINING/{self.setup}_hand_surface_filter_.pickle'

        with open(self.filterpath, 'rb') as handle:
                self.file_names = pickle.load(handle)

        if os.path.isfile(self.dataset_pkl_name) :

            with open(self.surfacefilterpath, 'rb') as handle:
                self.file_names = pickle.load(handle)

            print('file name loaded')

            with open(self.taxonomyfixpath, 'rb') as handle:
                self.taxonomyfixed = pickle.load(handle)

            print('taxonomy loaded')

            with open(self.dataset_pkl_name, 'rb') as handle:
                self.dict = pickle.load(handle)

            print('dataset dict loaded')

            with open(self.handoripath, 'rb') as handle:
                self.handori_dict = pickle.load(handle)

            print('hand data loaded')

            #self.file_names = self.dict['file_name']

        else :
            
            with open(self.filterpath, 'rb') as handle:
                self.file_names1 = pickle.load(handle)

            # /scratch/minjay/NIA_EXTRACT_FOR_TRAINING/train

            self.file_names2 = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')]

            file_name1 = set(self.file_names1)
            file_name2 = set(self.file_names2)

            self.file_names = list(file_name1 & file_name2)
            #self.file_names = [f for f in os.listdir(self.directory_path) if f.endswith('.pickle')][:1000]

            for file_name in tqdm.tqdm(self.file_names) :

                file = os.path.join(self.directory_path,file_name)

                with open(file, 'rb') as handle:
                    temp_dict = pickle.load(handle)

                #dict_keys(['taxonomy', 'obj_verts', 'obj_contact', 'hand_verts', 'hand_contact', 'hand_joints', 'obj_ids'])
                #           int         [3886, 3]    (3886, 1):nd.array [778, 3]  (778, 1):nd.array [21, 3]      '21': str

                self.dict[int(file_name.split('.')[0])] = temp_dict

            self.dict['file_name'] = self.file_names

            with open(self.dataset_pkl_name, 'wb') as handle:
                pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(self.taxonomyfixpath, 'rb') as handle:
                self.taxonomyfixed = pickle.load(handle)

            with open(self.handoripath, 'rb') as handle:
                self.handori_dict = pickle.load(handle)

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):

        sample = {}

        file_name = self.file_names[idx]

        ho_info    = self.dict[int(file_name.split('.')[0])]
        
        hand_dict = self.handori_dict[int(file_name.split('.')[0])]

        ho_gt = HandObject()
        
        obj_faces  = self.obj_faces_dict[int(ho_info['obj_ids'])]

        ho_gt.hand_verts   = ho_info['hand_verts'].to(self.device) #* 1000
        ho_gt.closed_faces = self.mano_faces
        ho_gt.obj_verts    = ho_info['obj_verts'].to(self.device)
        ho_gt.obj_faces    = obj_faces.to(self.device)
        ho_gt.hand_joints  = ho_info['hand_joints'].to(self.device)

        ho_gt.obj_contact  = ho_info['obj_contact']
        ho_gt.hand_contact  = ho_info['hand_contact']

        gt_contact = ho_info['hand_contact']
        
        gt_contact[SURFACE] = 0.05

        sample['taxonomy']     = TAX_DICT[int(self.taxonomyfixed[int(file_name.split('.')[0])])] - 1

        obj_sampled_idx = np.random.randint(0, len(ho_gt.obj_verts), 2048)

        hand_feats_gt, _ = ho_gt.generate_pointnet_features(obj_sampled_idx,self.device)

        sample['hand_feats_gt']   = hand_feats_gt
        
        mano_verts   = ho_gt.hand_verts

        sample['contact']         = gt_contact

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

        mano_verts = mano_verts - torch.Tensor([x_center, y_center, z_center]).to(self.device)

        sample['mano_verts'] = mano_verts

        return sample, idx  # Dummy label, replace as needed


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
parser.add_argument('--decay_step', type=int, default=2)
parser.add_argument('--weight_decay', type=float, default=0.0005)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=400)


# JoCoR parser
parser.add_argument('--jo_lr', type=float, default=0.001)
parser.add_argument('--jo_result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--jo_noise_rate', type=float, help='corruption rate, should be less than 1', default=0.5)
parser.add_argument('--jo_forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--jo_noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='pairflip')
parser.add_argument('--jo_num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--jo_exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--jo_n_epoch', type=int, default=200)
parser.add_argument('--jo_seed', type=int, default=1)
parser.add_argument('--jo_num_iter_per_epoch', type=int, default=100)
parser.add_argument('--jo_epoch_decay_start', type=int, default=80)
parser.add_argument('--jo_co_lambda', type=float, default=0.1)
parser.add_argument('--jo_adjust_lr', type=int, default=0)


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

# setup = 's0'
# split = 'train'

# dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, 'training_tsne_contact.pkl', flag_valid_obj=True)


# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# setup = 's0'
# split = 'test'

# dataset = NIADataset(setup,split,base_anno, base_source, seq_list, baseDir, device, 'testing_tsne_contact.pkl', flag_valid_obj=True)


#test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

train_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','train')

#train_dataset = ContactPose('/scratch/minjay/ContactOpt/data','train')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print('Train Loader Loaded')

test_dataset = CustomDataset('/scratch/minjay/NIA_EXTRACT_FOR_TRAINING','test')
#test_dataset = ContactPose('/scratch/minjay/ContactOpt/data','test')

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

print('Test Loader Loaded')

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

model1 = AE(args.in_channels,
           args.out_channels,
           args.output_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K).to(device)

model2 = AE(args.in_channels,
           args.out_channels,
           args.output_channels,
           args.latent_channels,
           edge_index_list,
           down_transform_list,
           up_transform_list,
           K=args.K).to(device)


#checkpoint = torch.load(f'/scratch/minjay/coma_refiner/out/{args.ori_exp_name}/checkpoints/checkpoint_300.pt')

# if 'new' not in args.exp_name :
#     print('############## pretrained_model_loaded #################')
#     model.load_state_dict(checkpoint['model_state_dict'])

testing_env = False

if testing_env :
    checkpoint = torch.load(f'/scratch/minjay/coma_taxonomy_prediction/out/classification_data_refine/checkpoints/checkpoint_1000.pt')

    print('############## pretrained_model_loaded #################')
    model.load_state_dict(checkpoint['model_state_dict'])

    #/scratch/minjay/coma_refiner/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 2

    #/scratch/minjay/coma/out/interpolation_exp_mano_contact[32, 32, 32, 64] 128 new both_none 



#train_eval.run_wo_contact(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

#train_eval.run_JoCoR(model1, model2, test_loader, test_loader, args.jo_n_epoch, writer, args.exp_name, device, args)

print(args)

train_eval.run_JoCoR(model1, model2, train_loader, test_loader, args.jo_n_epoch, writer, args.exp_name, device, args)

#train_eval.run_tester(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

#train_eval.run_conditioned_taxonomy(model, train_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)

#train_eval.run_conditioned_taxonomy(model, test_loader, test_loader, args.epochs, optimizer, scheduler, writer, args.exp_name, device, args)
