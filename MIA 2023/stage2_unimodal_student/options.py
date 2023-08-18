import argparse
import os

import torch

### Parser

def parse_args():
    parser = argparse.ArgumentParser()
    ### arguments for t-SVD
    parser.add_argument('--tSVD_mode', type=str, default="path", help="[path, omic, pathomic]")
    parser.add_argument('--tSVD_loss', type=str, default="False")
    parser.add_argument('--n_views', type=int, default=4, help='number of views for tSVD constraint')
    parser.add_argument('--Lambda_global', type=float, default=0.05, metavar='N',
                    help='the trade-off parameter of losses')
    parser.add_argument('--mu', type=float, default=1e-5, metavar='N',
                        help='the scalar mu')
    parser.add_argument('--max_mu', type=float, default=1, metavar='N',
                        help='the maximum of mu')
    parser.add_argument('--pho', type=float, default=1.1, metavar='N',
                        help='the scalar pho')
    parser.add_argument('--aux_iter', type=int, default=1, metavar='N',
                        help='when to update auxiliary variable')
    parser.add_argument('--proto_beta', type=float, default=0.5, metavar='N',
                        help='moving weight for updating the prototypes')

    parser.add_argument('--orth_loss', type=str, default="False", 
        help='whether to regularize the multi-modal feature to be orthogonal.')
    parser.add_argument('--student_customize', type=str, default="False", 
        help='whether mask the KD loss according to the similarity of KD gradient and CE loss gradient.')
    parser.add_argument('--assign_weights', type=str, default="False", 
        help='whether assign weights to different KD losses according to the gradient similarity.')
    parser.add_argument('--distill', type=str, default='kd', 
                        choices=['kd', 'feats_KL', 'hint', 'attention', 'similarity','correlation', 'vid', 
                        'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--kd_T', type=float, default=1, help='temperature for KD distillation')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    
    parser.add_argument('--cut_fuse_grad', default=False, action="store_true", 
        help='whether cut the gradients from the fuse branch to single modality.')
    parser.add_argument('--select_pos_mode', type=str, default='random', help='the rule to select positive pairs for CRD')
    parser.add_argument('--select_pos_pairs', default=True, action="store_true", 
        help='whether to select positive pairs for the CRD loss.')
    parser.add_argument('--select_neg_pairs', type=str, default="True", 
        help='whether to select negative pairs for the CRD loss.')
    parser.add_argument('--CE_grads', default=False, action="store_true", 
        help='whether use the gradients of CE loss to assign teacher weights.')
    # parser.add_argument('--fixed_model', type=str, default='pathomic_self_MT_KD', help='mode')
    parser.add_argument('--fixed_model', type=str, default='1023_pathomic_MT', help='mode')
    # parser.add_argument('--fixed_model', type=str, default='0322_pofusion_path_omic_4views_tsvd_lam0.1', help='mode')

    parser.add_argument('--svm_norm', default=False, action="store_true", help='if use norm when compute with svm')
    parser.add_argument('--grad_place', type=str,  default='feat', help='where to compare gradients.')

    ### The method to transform the omic data. xxh, 2022/01/01
    parser.add_argument('--omic_transform', type=str,  default='drop', help='[drop, vime]')

    ### whether return feature gradients. xxh, 2021/12/16
    parser.add_argument('--return_grad', type=str,  default='False', help='whether to return gradients.')

    ### for knowledge distillation
    parser.add_argument('--start_KD', type=int, default=10, help='which epoch start to employ KD in the model')
    parser.add_argument('--pred_distill', type=int, default=1, help='whether to use pred KD loss')
    parser.add_argument('--num_teachers', type=int, default=1, help='number of teacher for each single modality')
    parser.add_argument('--KD_weight', type=float, default=1.0, help='the weight for the KD loss')
    parser.add_argument('--KD_type', type=str,  default='KD', help='[KD, CRD, CRD_KD].')
    parser.add_argument('--sample_KD', type=str,  default='False', help='whether to select some samples for KD.')
    parser.add_argument('--global_step', type=int,  default=0, help='global_step')
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_rampup', type=float,  default=10, help='consistency_rampup')
    parser.add_argument('--which_teacher', type=str,  default='fuse', 
        help='[fuse, self_EMA], when num_teachers=1, choose one teacher for distillation')

    ### for CRD loss
    parser.add_argument('--CRD_distill', type=int, default=1, help='whether use the CRD loss')
    parser.add_argument('--CRD_mode', type=str, default="sup", choices=['sup', 'unsup'])
    parser.add_argument('--CRD_weight', type=float, default=0.1, help='the weight for the SP loss')

    parser.add_argument('--s_dim', type=int, default=128, help='feature dim of the student model')
    parser.add_argument('--t_dim', type=int, default=128, help='feature dim of the EMA teacher')
    parser.add_argument('--feat_dim', type=int, default=128, help='reduced feature dimension')
    parser.add_argument('--pos_mode', default='multi_pos', type=str, choices=['exact', 'relax', 'multi_pos'])
    parser.add_argument('--nce_p', default=300, type=int, help='number of positive samples for NCE')
    parser.add_argument('--nce_p2', default=10, type=int, help='number of positive samples for NCE')
    parser.add_argument('--nce_k', default=700, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_k2', default=512, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--n_data', default=1024, type=int, help='number of training samples')
    parser.add_argument('--neg_mode', default='all_others', type=str, choices=['all_others', 'diff_class', 'both_models'])

    ### for similarity-preserving (SP) loss
    parser.add_argument('--SP_distill', type=int, default=0, help='whether to use SP loss')
    parser.add_argument('--SP_weight', type=float, default=1.0, help='the weight for the SP loss')

    ### for supervised contrastive loss
    parser.add_argument('--supcon_distill', type=int, default=0, help='whether to use supcon loss')
    parser.add_argument('--supcon_weight', type=float, default=1.0, help='the weight for the supcon loss')

    ### common params
    # parser.add_argument('--dataroot', default='./data/TCGA_GBMLGG', help="datasets")
    parser.add_argument('--dataroot', default='../pathomic_fusion_20221023_miccai/data/TCGA_GBMLGG', help="datasets")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='grad_15', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='pathomic', help='mode')
    parser.add_argument('--model_name', type=str, default='omic', help='mode')
    parser.add_argument('--use_vgg_features', type=int, default=0, help='Use pretrained embeddings')
    parser.add_argument('--use_rnaseq', type=int, default=0, help='Use RNAseq data.')
    
    parser.add_argument('--task', type=str, default='grad', help='surv | grad')
    parser.add_argument('--useRNA', type=int, default=0) # Doesn't work at the moment...:(
    parser.add_argument('--useSN', type=int, default=1)
    parser.add_argument('--act_type', type=str, default='LSM', help='activation function')
    parser.add_argument('--input_size_omic', type=int, default=80, help="input_size for omic vector")
    parser.add_argument('--input_size_path', type=int, default=512, help="input_size for path images")
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--save_at', type=int, default=20, help="adsfasdf")
    parser.add_argument('--label_dim', type=int, default=3, help='size of output')
    parser.add_argument('--measure', default=1, type=int, help='disables measure while training (make program faster)')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=0, type=int)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--beta2', type=float, default=0.999, help='0.9, 0.5 | 0.25 | 0')
    parser.add_argument('--lr_policy', default='linear', type=str, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--lr_decay_iters', default=10, type=int, help='decay lr after 20 epochs')
    parser.add_argument('--finetune', default=1, type=int, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--final_lr', default=0.1, type=float, help='Used for AdaBound')
    parser.add_argument('--reg_type', default='omic', type=str, help="regularization type")
    parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='start of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help="Number of batches to train/test for. Default: 256")

    parser.add_argument('--lambda_cox', type=float, default=1)
    parser.add_argument('--lambda_reg', type=float, default=3e-4)
    parser.add_argument('--lambda_nll', type=float, default=1)

    parser.add_argument('--fusion_type', type=str, default="pofusion", help='concat|pofusion|LMF|HFB|GPDBN|mmdynamics')
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--use_bilinear', type=int, default=1)
    parser.add_argument('--path_gate', type=int, default=1)
    parser.add_argument('--omic_gate', type=int, default=1)
    parser.add_argument('--path_dim', type=int, default=128)
    parser.add_argument('--omic_dim', type=int, default=128)
    parser.add_argument('--path_scale', type=int, default=1)
    parser.add_argument('--omic_scale', type=int, default=1)
    parser.add_argument('--mmhid', type=int, default=128)

    parser.add_argument('--init_type', type=str, default='max', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')
    parser.add_argument('--use_edges', default=1, type=float, help='Using edge_attr')
    parser.add_argument('--pooling_ratio', default=0.2, type=float, help='pooling ratio for SAGPOOl')
    parser.add_argument('--lr', default=0.0005, type=float, help='5e-4 for Adam | 1e-3 for AdaBound')
    parser.add_argument('--weight_decay', default=4e-4, type=float, help='Used for Adam. L2 Regularization on weights. I normally turn this off if I am using L1. You should try')
    parser.add_argument('--GNN', default='GCN', type=str, help='GCN | GAT | SAG. graph conv mode for pooling')
    parser.add_argument('--patience', default=0.005, type=float)
    
    ##########meilu
    parser.add_argument('--num_superpixels', type=int, default=100, help='the number of superpixels')    
    parser.add_argument('--Path_K', type=int, default=5, help='the number of selected superpixels')    
    parser.add_argument('--Omic_K', type=int, default=5, help='the number of selected superpixels')    

    parser.add_argument('--start_epoch', type=int, default=1, help='starting point')        
    parser.add_argument('--masking', type=int, default=0, help='whether introduce masking')    

    
    
    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
