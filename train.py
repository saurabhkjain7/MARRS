from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
from utils.utils import *
from utils.eval import *
from utils.features import *
from utils.return_dataset import return_dataset
from utils.lr_schedule import inv_lr_scheduler
from scipy.linalg import sqrtm
from datetime import datetime
from torch.autograd import Variable
import copy

# training settings
parser = argparse.ArgumentParser(description='Pytorch SSDA MARRS')
parser.add_argument('--seed', 
                    type=int, 
                    default=12345, 
                    metavar='S',
                    help='seed to control randomness in network')
parser.add_argument('-g', 
                    '--gpu_id', 
                    type=str, 
                    default='0',
                    help='gpu id')       
parser.add_argument('--dataset', 
                    type=str, 
                    choices=['multi','office_home','office'],
                    default='multi', 
                    help='type of dataset')
parser.add_argument('--shots', 
                    type=int, 
                    default=3, 
                    help='Number of labeled samples per class from target domain')
parser.add_argument('--aug1', 
                    type=str, 
                    choices=['none','perspective','randaument'],
                    default='perspective', 
                    help='type of augmentation for convnext backbone')
parser.add_argument('--aug2', 
                    type=str, 
                    choices=['none','perspective','randaument'],
                    default='none', 
                    help='type of augmentation for swin backbone')
parser.add_argument('--coral1', 
                    action='store_true', 
                    default=False,
                    help='apply coral in convnext backbone or not')
parser.add_argument('--coral2', 
                    action='store_true', 
                    default=False,
                    help='apply coral in swin backbone or not')
parser.add_argument('--kd', 
                    action='store_true', 
                    default=False,
                    help='apply knowledge distillation or not')
parser.add_argument('--epsilon', 
                    type=float, 
                    default=0.1, 
                    help='smoothness parameter in label regularization')
parser.add_argument('--lr', 
                    type=float, 
                    default=30, 
                    help='learning rate')
parser.add_argument('--multi', 
                    type=float, 
                    default=0.1, 
                    metavar='MLT',
                    help='learning rate multiplication')   
parser.add_argument('--tau_s', 
                    type=float, 
                    default=0.8, 
                    help='threshold for source data')
parser.add_argument('--tau_tu', 
                    type=float, 
                    default=0.9, 
                    help='threshold for target unlabeled data')
parser.add_argument('--tau_tu_kd', 
                    type=float, 
                    default=0.7, 
                    help='threshold for target unlabeled data during student model training')
parser.add_argument('--n_outer',
                    type=int, 
                    default=30,
                    help='number of outer iterations in main classifier training')
parser.add_argument('--n_inner',
                    type=int, 
                    default=400,
                    help='number of inner iterations in main classifier training')
parser.add_argument('--n_max',
                    type=int, 
                    default=400,
                    help='number of iterations in initial classifier training')
parser.add_argument('--lambda_s_ini', 
                    type=float,
                    default=0.4, 
                    help='scaling parameter for source data loss in initial classifier training') 
parser.add_argument('--lambda_s', 
                    type=float, 
                    default=0.1, 
                    help='scaling parameter for source data loss in main classifier training')
parser.add_argument('--lambda_s_kd', 
                    type=float, 
                    default=0.1, 
                    help='scaling parameter for source data loss in student training')
parser.add_argument('--lambda_tl_ini', 
                    type=float, 
                    default=0.2, 
                    help='scaling parameter for target labeled data loss in initial classifier training') 
parser.add_argument('--lambda_tl', 
                    type=float, 
                    default=0.05, 
                    help='scaling parameter for target labeled data loss in main classifier training')
parser.add_argument('--lambda_tl_kd', 
                    type=float, 
                    default=0.05, 
                    help='scaling parameter for target labeled data loss in student training')
parser.add_argument('--lambda_co', 
                    type=float, 
                    default=0.9,
                    help='scaling parameter for target unlabeled data in co-training loss')
parser.add_argument('--lambda_tu_kd', 
                    type=float, 
                    default=0.9, 
                    help='scaling parameter for target unlabeled data in student training')
parser.add_argument('--lambda_cons', 
                    type=float, 
                    default=0.5, 
                    help='scaling parameter for target unlabeled data in consistency loss')
parser.add_argument('--distill_steps', 
                    type=int, 
                    default=11000, 
                    help='number of steps in student model training')


args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id    
torch.cuda.empty_cache()
set_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#####################  Main training algorithm of classifiers  #####################  
def train(D_C, x_s_c, x_t_c, x_tu_c, x_tu_c_aug, \
             D_S, x_s_s, x_t_s, x_tu_s, x_tu_s_aug, \
             x_tv_c, x_tv_s, y_s, y_t, y_tv, y_tu, device='cuda'):

    ### initializing optimizers
    D_C.to(device)
    D_S.to(device)
    optimizer_d_c = optim.SGD(D_C.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0000, nesterov=True)
    optimizer_d_s = optim.SGD(D_S.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0000, nesterov=True)

    ### creating tuples to store state of classifiers (target_acc, source_acc, val_acc, unlabeled_predictions)
    return_list_C = []
    cur_return_list_C = []
    return_list_S = []
    cur_return_list_S = []

    ### initializing some important variables
    best_val_acc_c = 0
    best_test_acc_c = 0
    best_val_acc_s = 0
    best_test_acc_s = 0
    curr_target_thresh=args.tau_tu

    ### training starts (outer loop)
    for iter in range(args.n_outer):

        ### updating threshold for target unlabeled data
        if iter>0 and iter%10 == 0:
            curr_target_thresh=curr_target_thresh-0.1
         
        ## calculating masks and pseudo labels using ConvNext based classifier 
        unlabeled_confidence_c, unlabeled_preds_c = torch.max(F.softmax(D_C(x_tu_c.to(device)), -1), -1)
        source_confidence_c, _ = torch.max(F.softmax(D_C(x_s_c.to(device)), -1), -1)
        source_mask_c = source_confidence_c > args.tau_s
        unlabeled_mask_c = unlabeled_confidence_c > curr_target_thresh  
        pseudo_labels_c = unlabeled_preds_c.detach()

        ## calculating masks and pseudo labels using Swin based classifier
        unlabeled_confidence_s, unlabeled_preds_s = torch.max(F.softmax(D_S(x_tu_s.to(device)), -1), -1)
        source_confidence_s, _ = torch.max(F.softmax(D_S(x_s_s.to(device)), -1), -1)
        source_mask_s = source_confidence_s > args.tau_s
        unlabeled_mask_s = unlabeled_confidence_s > curr_target_thresh 
        pseudo_labels_s = unlabeled_preds_s.detach()

        ### inner loop
        for _ in range(args.n_inner):
            
            ############# training of convnext based classifier ##############

            ### calculating outputs 
            y_hat_s_c = D_C(x_s_c)
            y_hat_t_c = D_C(x_t_c)
            y_hat_tu_c = D_C(x_tu_c)
            y_hat_tu_c_aug = D_C(x_tu_c_aug)

            ### label smoothing regularization loss on labeled data
            loss = args.lambda_s * (CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.epsilon)(y_hat_s_c[source_mask_c],y_s[source_mask_c],T=1))
            loss += args.lambda_tl * (CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.epsilon)(y_hat_t_c,y_t,T=1))

            ### consistency regularization loss on target unlabeled data
            loss +=  args.lambda_cons*(F.cross_entropy(y_hat_tu_c_aug, pseudo_labels_c, reduction='none')*unlabeled_mask_c).mean()

            ### co-training loss on target unlabeled data
            loss += args.lambda_co*(F.cross_entropy(y_hat_tu_c, pseudo_labels_s, reduction='none')*unlabeled_mask_s).mean()
            
            ### updating parameters
            optimizer_d_c.zero_grad()
            loss.backward()
            optimizer_d_c.step()

            ############# training of swin based classifier ##############

            ### calculating outputs 
            y_hat_s_s = D_S(x_s_s)
            y_hat_t_s = D_S(x_t_s)
            y_hat_tu_s = D_S(x_tu_s)
            y_hat_tu_s_aug = D_S(x_tu_s_aug)

            ## label smoothing regularization loss on labeled data
            loss = args.lambda_s * (CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.epsilon)(y_hat_s_s[source_mask_s],y_s[source_mask_s],T=1))
            loss += args.lambda_tl * (CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=args.epsilon)(y_hat_t_s,y_t,T=1))

            ## consistency regularization loss on target unlabeled data
            loss +=  args.lambda_cons*(F.cross_entropy(y_hat_tu_s_aug, pseudo_labels_s, reduction='none')*unlabeled_mask_s).mean()
            
            ## co-training loss on target unlabeled data
            loss += args.lambda_co*(F.cross_entropy(y_hat_tu_s, pseudo_labels_c, reduction='none')*unlabeled_mask_c).mean()
           
            ### updating parameters
            optimizer_d_s.zero_grad()
            loss.backward()
            optimizer_d_s.step()
        
        if iter>0 and (iter+1)%5 == 0:

            print('{} iteration of outer training ...'.format(iter+1))
            cur_return_list_C = get_accs_and_labels(D_C, x_s_c, x_tv_c, x_tu_c, y_s, y_tv, y_tu)
            cur_return_list_S = get_accs_and_labels(D_S, x_s_s, x_tv_s, x_tu_s, y_s, y_tv, y_tu)
        
            if cur_return_list_C[2] >= best_val_acc_c :
                best_val_acc_c = cur_return_list_C[2]
                best_test_acc_c = cur_return_list_C[0]
                best_D_C = copy.deepcopy(D_C)
                return_list_C = []
                return_list_C.append(cur_return_list_C)

            if cur_return_list_S[2] >= best_val_acc_s :
                best_val_acc_s = cur_return_list_S[2]
                best_test_acc_s = cur_return_list_S[0]
                best_D_S = copy.deepcopy(D_S)
                return_list_S = []
                return_list_S.append(cur_return_list_S)
            
            print("For conv: curr target acc: {:.1f} , curr val acc: {:.1f} , best target acc: {:.1f}, best val acc: {:.1f}".format(cur_return_list_C[0],cur_return_list_C[2],best_test_acc_c,best_val_acc_c))
            print("For swin: curr target acc: {:.1f} , curr val acc: {:.1f} , best target acc: {:.1f}, best val acc: {:.1f}".format(cur_return_list_S[0],cur_return_list_S[2],best_test_acc_s,best_val_acc_s))

    return return_list_C, best_D_C, return_list_S, best_D_S

#####################  Initial training algorithm of classifiers  #####################  
def train_initial(inc, x_s, x_t, y_s, y_t, num_classes, device='cuda'):
    D = nn.Linear(inc, num_classes, bias=False).to(device)  
    D.to(device)
    optimizer_d = optim.SGD(D.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.000, nesterov=True)

    for _ in range(args.n_max):
        y_hat_s = D(x_s)
        y_hat_t = D(x_t)
        loss = args.lambda_s_ini*F.cross_entropy(y_hat_s, y_s)
        loss += args.lambda_tl_ini*F.cross_entropy(y_hat_t, y_t)
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
    return D

#####################  Training algorithm of student model  #####################  
def train_student(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, params, source_domain, target_domain,D_C,D_S,wandb=None):

    ### Defining variables to hold data 
    im_data_s = torch.FloatTensor(1)
    im_data_t = torch.FloatTensor(1)
    im_data_tu = torch.FloatTensor(1)
    gt_labels_s = torch.LongTensor(1)
    gt_labels_t = torch.LongTensor(1)
   
    im_data_s = im_data_s.cuda()
    im_data_t = im_data_t.cuda()
    im_data_tu = im_data_tu.cuda() 
    gt_labels_s = gt_labels_s.cuda()
    gt_labels_t = gt_labels_t.cuda()


    im_data_s = Variable(im_data_s)
    im_data_t = Variable(im_data_t)
    im_data_tu = Variable(im_data_tu)
    gt_labels_s = Variable(gt_labels_s)
    gt_labels_t = Variable(gt_labels_t)
   
    net_G.train()

    ### Setting optimizer
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    def zero_grad_all():
        optimizer_g.zero_grad()
   

    ### Setting important variables
    criterion = nn.CrossEntropyLoss().cuda()
    best_acc_val, best_acc_test, cur_acc_test = 0, 0, 0
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    target_unl_iter = iter(target_loader_unl)

    ### Training starts from here
    for step in range(args.distill_steps):

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        lr = optimizer_g.param_groups[0]['lr']

        try:
            data_batch_source = source_iter.next()
        except:
            source_iter = iter(source_loader)
            data_batch_source = source_iter.next()

        try:
            data_batch_target = target_iter.next()
        except:
            target_iter = iter(target_loader)
            data_batch_target = target_iter.next()

        try:
            data_batch_unl = target_unl_iter.next()
        except:
            target_unl_iter = iter(target_loader_unl)
            data_batch_unl = target_unl_iter.next()


        im_data_s.resize_(data_batch_source[0].size()).copy_(data_batch_source[0])
        gt_labels_s.resize_(data_batch_source[1].size()).copy_(data_batch_source[1])
        im_data_t.resize_(data_batch_target[0].size()).copy_(data_batch_target[0])
        gt_labels_t.resize_(data_batch_target[1].size()).copy_(data_batch_target[1])
        im_data_tu.resize_(data_batch_unl[0].size()).copy_(data_batch_unl[0])
        ind_data_tu = data_batch_unl[2]

        ### Calculating pseudo labels for unlabeled data from pretrained network 
        features_unl = get_features_unlabeled(target_domain,ind_data_tu,augmentation = args.aug1,network = 'convnext_xlarge_384_in22ft1k',dataset=args.dataset)
        features_unl = features_unl.to(device)
        ps_labels_t = F.softmax(D_C(features_unl), -1)
        ps_prob_t_c, ps_labels_t_c = torch.max(ps_labels_t, -1)

        features_unl = get_features_unlabeled(target_domain,ind_data_tu,augmentation = args.aug2,network = 'swin_large_patch4_window12_384',dataset=args.dataset)
        features_unl = features_unl.to(device)
        ps_labels_t = F.softmax(D_S(features_unl), -1)
        ps_prob_t_s, ps_labels_t_s = torch.max(ps_labels_t, -1)
    
        zero_grad_all()

        ### Passing labeled data through network
        data = torch.cat((im_data_s, im_data_t), 0)
        output_lab = net_G(data)
        output_lab_s, output_lab_t = output_lab.chunk(2)

        ### Calculating losses on labeled data
        loss_s = criterion(output_lab_s,gt_labels_s)
        loss_t = criterion(output_lab_t,gt_labels_t) 
      
        ### Calculating loss on unlabeled data
        ps_mask = (ps_prob_t_s > args.tau_tu_kd) & (ps_prob_t_c > args.tau_tu_kd) & (ps_labels_t_c == ps_labels_t_s)
        im_data_tu = im_data_tu[ps_mask]
        ps_labels_t = ps_labels_t_s[ps_mask]

        output1 = net_G(im_data_tu)
        loss_tu = criterion(output1,ps_labels_t)

        ### Combining losses
        loss = args.lambda_s_kd*loss_s + args.lambda_tl_kd*loss_t + args.lambda_tu_kd*loss_tu

        ### Updating parameters
        loss.backward()
        optimizer_g.step()
        zero_grad_all()

        log_train = ' Train Ep: {} Lr: {} Loss: {:.5f} Loss_s: {:.5f} Loss_t: {:.5f} Loss_tu: {:.5f}\n '.\
            format(step, lr, loss, loss_s, loss_t, loss_tu)

        if (step % 500 == 0) and step > 0:
            print(log_train)
            
        if (step % 500 == 0) and step > 0:
            cur_acc_val = test(target_loader_val, net_G)
            cur_acc_test = test(target_loader_test, net_G)
            net_G.train()
            
            if cur_acc_val > best_acc_val:
                best_acc_val = cur_acc_val
                best_acc_test = cur_acc_test

            log_train = ' Current_acc_val: {:.5f} Current_acc_test: {:.5f} Best_acc_val: {:.5f} Best_acc_test: {:.5f}  \n '.\
            format(cur_acc_val, cur_acc_test, best_acc_val, best_acc_test)
            print(log_train)

    resnet_pred[(source_domain,target_domain)] = (best_acc_test,cur_acc_test,best_acc_val)

#####################  Anchor function  ##################### 
def get_results(source, target, dataset, shots, num_classes, device):
    ### Get already computed features from convnext backbone
    source_features_c, source_labels, val_target_features_c, val_target_labels, \
    unlabeled_target_features_c, unlabeled_target_labels, labeled_target_features_c, \
    labeled_target_labels = get_features(args.aug1 , 'convnext_xlarge_384_in22ft1k', dataset, source, target, shots)

    _, _ , _ , _ , unlabeled_target_features_c_aug,\
    _ , _ , _ = get_features('randaugment', 'convnext_xlarge_384_in22ft1k', dataset, source, target, shots)

    if args.coral1:
        x_tu_c, x_t_c, x_s_c, x_tv_c = coral(source_features_c, val_target_features_c, unlabeled_target_features_c, labeled_target_features_c)
        x_tu_c_aug = unlabeled_target_features_c_aug
    else:
        x_tu_c, x_t_c, x_s_c, x_tu_c_aug, x_tv_c = unlabeled_target_features_c, labeled_target_features_c, source_features_c, \
                                                    unlabeled_target_features_c_aug, val_target_features_c
        
    ### Get already computed features from swin backbone
    source_features_s, source_labels, val_target_features_s, val_target_labels, \
    unlabeled_target_features_s, unlabeled_target_labels, labeled_target_features_s, \
    labeled_target_labels = get_features(args.aug2 , 'swin_large_patch4_window12_384', dataset, source, target, shots)
    
    _, _ , _ , _ , unlabeled_target_features_s_aug, \
    _ , _ , _ = get_features('randaugment', 'swin_large_patch4_window12_384', dataset, source, target, shots)
    
    if args.coral2:
        x_tu_s, x_t_s, x_s_s, x_tv_s = coral(source_features_s, val_target_features_s, unlabeled_target_features_s, labeled_target_features_s)
        x_tu_s_aug = unlabeled_target_features_s_aug
    else:
        x_tu_s, x_t_s, x_s_s, x_tu_s_aug, x_tv_s = unlabeled_target_features_s, labeled_target_features_s, source_features_s, \
                                                    unlabeled_target_features_s_aug, val_target_features_s

    ### Normalizing the output features to surface of L2 ball and assigning the device
    x_s_s = F.normalize(x_s_s).to(device) 
    x_t_s = F.normalize(x_t_s).to(device) 
    x_tu_s = F.normalize(x_tu_s).to(device) 
    x_tv_s = F.normalize(x_tv_s).to(device) 
    x_tu_s_aug = F.normalize(x_tu_s_aug).to(device) 

    x_s_c = F.normalize(x_s_c).to(device) 
    x_t_c = F.normalize(x_t_c).to(device) 
    x_tu_c = F.normalize(x_tu_c).to(device) 
    x_tv_c = F.normalize(x_tv_c).to(device) 
    x_tu_c_aug = F.normalize(x_tu_c_aug).to(device) 

    ### Assigning the device to the labels 
    y_s = source_labels.long().to(device)
    y_t = labeled_target_labels.long().to(device)
    y_tu = unlabeled_target_labels.long().to(device)
    y_tv = val_target_labels.long().to(device)
  
    
    print('Initial training of convnext based classifier on labeled data ... ' )
    D_C = train_initial(2048, x_s_c, x_t_c, y_s, y_t, num_classes, device) 
  
    print('Initial training of swin based classifier on labeled data ...')
    D_S = train_initial(1536, x_s_s, x_t_s, y_s, y_t, num_classes, device) 
 
    return_list_C, best_D_C , return_list_S, best_D_S = train(D_C, x_s_c, x_t_c, x_tu_c,x_tu_c_aug, \
                                                                 D_S, x_s_s, x_t_s, x_tu_s,x_tu_s_aug, \
                                                                x_tv_c,x_tv_s,y_s, y_t, y_tv, y_tu) 

    ### checking whether student model results are required or not
    if args.kd:
        source_loader, target_loader, target_loader_unl, target_loader_val, \
        target_loader_test, _ = return_dataset(source, target, args=args, return_idx=False)

        ### defining student network
        net_G = models.mobilenet_v2(pretrained=True)
        num_ftrs = net_G.classifier[1].in_features
        net_G.classifier[1] = nn.Linear(num_ftrs, num_classes)

        ### setting learning rate parameters for scheduling
        params = []
        for key, value in dict(net_G.named_parameters()).items():
            if value.requires_grad:
                if 'classifier' not in key:
                    params += [{'params': [value], 'lr': args.multi,
                                'weight_decay': 0.0005}]
                else:
                    params += [{'params': [value], 'lr': args.multi * 10,
                                'weight_decay': 0.0005}]
                    
        net_G = torch.nn.DataParallel(net_G).cuda()
        print('Student model training starts ... ' )
        train_student(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,\
                            net_G, params, source, target,best_D_C,best_D_S)

    return  return_list_C, best_D_C.cpu(), x_tu_c.detach().cpu(), x_tv_c.detach().cpu(), \
                return_list_S, best_D_S.cpu(), x_tu_s.detach().cpu(), x_tv_s.detach().cpu()


#####################  Code starts from here  #####################  
if args.dataset == 'multi':
    domain_pairs = [('real','clipart'),
                   ('real','painting'),
                   ('painting','clipart'),
                   ('clipart', 'sketch'),
                   ('sketch', 'painting'),
                   ('real', 'sketch'),
                   ('painting','real')]
    num_classes = 126

elif args.dataset == 'office_home':
    domain_pairs = [('Real','Clipart'),
                   ('Real','Product'),
                   ('Real','Art'),
                   ('Product', 'Real'),
                   ('Product', 'Clipart'),
                   ('Product', 'Art'),
                   ('Art','Product'),
                   ('Art','Clipart'),
                   ('Art','Real'),
                   ('Clipart','Real'),
                   ('Clipart','Art'),
                   ('Clipart','Product')]
    num_classes = 65

else:
    assert args.dataset == 'office'
    domain_pairs = [('dslr','amazon'),
                    ('webcam','amazon')]
    num_classes = 31


dic_c = {}    # stores predictions of convnext based classifier
dic_D_c = {}  # stores weights of convnext based classifier
dic_s = {}    # stores predictions of swin based classifier
dic_D_s = {}  # stores weights of swin based classifier 
resnet_pred={}

start_time = datetime.now()   

for source, target in domain_pairs:
    print('================================================================================')
    print('Training {} shots on {} for {} to {}'.format(args.shots,args.dataset,source,target))
    print('================================================================================')
    rl_C, D_C, x_tu_c, x_tv_c, rl_S, D_S, x_tu_s, x_tv_s = get_results( source, target, args.dataset, args.shots, num_classes, device)

    ## Storing predictions of convnext
    dic_c[('convnext_xlarge_384_in22ft1k', source, target)] = rl_C  
    dic_D_c[('convnext_xlarge_384_in22ft1k', source, target)] = (D_C, x_tu_c, x_tv_c) 

    ## Storing predictions of swin
    dic_s[('swin_large_patch4_window12_384', source, target)] = rl_S 
    dic_D_s[('swin_large_patch4_window12_384', source, target)] = (D_S, x_tu_s, x_tv_s) 

print()
print()
print()

print('================================================================================')
print('Final test Results(model 1) of ',args.shots," shots on ",args.dataset)
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print('mean')
print()
avg = 0
for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(args.dataset, target, num=args.shots)
    y_hat_sum = torch.zeros((unlabeled_target_labels.shape[0], num_classes)).to(device)

    ### For convnext
    D_C, x_tu_c, x_tv_c = dic_D_c[('convnext_xlarge_384_in22ft1k', source, target)] # retrieving classifier weights
    D_C = D_C.to(device)
    x_tu_c = x_tu_c.to(device)
    y_hat1 = F.softmax(D_C(x_tu_c), -1) # calculating prediction
    y_hat_sum = y_hat_sum + y_hat1  # adding prediction to sum


    _, unlabeled_preds = torch.max(y_hat_sum, -1) # calculating label corresponding to maximum probability

    ps_lab = unlabeled_preds
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    avg += acc
    print('{:.1f}'.format(acc), end = ', ')

if args.dataset == 'office_home':
    avg = avg / 12
    print('{:.1f}'.format(avg)) 
elif args.dataset == 'multi':
    avg = avg / 7
    print('{:.1f}'.format(avg)) 
else:
    avg = avg / 2
    print('{:.1f}'.format(avg)) 


print()
print()
print()

print('================================================================================')
print('Final test Results(model 2) of ',args.shots," shots on ",args.dataset)
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print('mean')
print()
avg = 0
for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(args.dataset, target, num=args.shots)
    y_hat_sum = torch.zeros((unlabeled_target_labels.shape[0], num_classes)).to(device)


    ## For swin
    D_S, x_tu_s, x_tv_s = dic_D_s[('swin_large_patch4_window12_384', source, target)] # retrieving classifier weights
    D_S = D_S.to(device)
    x_tu_s = x_tu_s.to(device)
    y_hat2 = F.softmax(D_S(x_tu_s), -1) # calculating prediction
    y_hat_sum = y_hat_sum + y_hat2  # adding prediction to sum

    _, unlabeled_preds = torch.max(y_hat_sum, -1) # calculating label corresponding to maximum probability

    ps_lab = unlabeled_preds
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    avg += acc
    print('{:.1f}'.format(acc), end = ', ')

if args.dataset == 'office_home':
    avg = avg / 12
    print('{:.1f}'.format(avg)) 
elif args.dataset == 'multi':
    avg = avg / 7
    print('{:.1f}'.format(avg)) 
else:
    avg = avg / 2
    print('{:.1f}'.format(avg)) 

print()
print()
print()

print('================================================================================')
print('Final test Results(ensemble) of ',args.shots," shots on ",args.dataset)
print('================================================================================')
for source, target in domain_pairs:
    print('{}->{}'.format(source[0], target[0]), end = ', ')
print('mean')
print()
avg = 0
for source, target in domain_pairs:
    unlabeled_target_labels = get_unlabeled_target_labels(args.dataset, target, num=args.shots)
    y_hat_sum = torch.zeros((unlabeled_target_labels.shape[0], num_classes)).to(device)

    ## For convnext
    D_C, x_tu_c, x_tv_c = dic_D_c[('convnext_xlarge_384_in22ft1k', source, target)] # retrieving classifier weights
    D_C = D_C.to(device)
    x_tu_c = x_tu_c.to(device)
    y_hat1 = F.softmax(D_C(x_tu_c), -1) # calculating prediction
    y_hat_sum = y_hat_sum + y_hat1  # adding prediction to sum

    ## For swin
    D_S, x_tu_s, x_tv_s = dic_D_s[('swin_large_patch4_window12_384', source, target)] # retrieving classifier weights
    D_S = D_S.to(device)
    x_tu_s = x_tu_s.to(device)
    y_hat2 = F.softmax(D_S(x_tu_s), -1) # calculating prediction
    y_hat_sum = y_hat_sum + y_hat2  # adding prediction to sum

    _, unlabeled_preds = torch.max(y_hat_sum, -1) # calculating label corresponding to maximum probability

    ps_lab = unlabeled_preds
    prediction = unlabeled_preds.cpu()
    acc = ((prediction == unlabeled_target_labels).sum()/len(prediction)).item()*100.
    avg += acc
    print('{:.1f}'.format(acc), end = ', ')

if args.dataset == 'office_home':
    avg = avg / 12
    print('{:.1f}'.format(avg)) 
elif args.dataset == 'multi':
    avg = avg / 7
    print('{:.1f}'.format(avg)) 
else:
    avg = avg / 2
    print('{:.1f}'.format(avg)) 

if args.kd:
    print()
    print()
    print('================================================================================')
    print('Final Results (ResNet-34) in ',args.shots," shots on ",args.dataset)
    print('================================================================================')
    for source, target in domain_pairs:
        print('{}->{}'.format(source[0], target[0]), end = ', ')
    print('mean')
    print()

    avg = 0
    for source, target in domain_pairs:
        acc= resnet_pred[(source,target)][0]

        print("{:.1f}".format(acc), end= ', ')
        avg += acc

    if args.dataset == 'office_home':
        avg = avg / 12
        print('{:.1f}'.format(avg)) 
    elif args.dataset == 'multi':
        avg = avg / 7
        print('{:.1f}'.format(avg)) 
    else:
        avg = avg / 2
        print('{:.1f}'.format(avg)) 

print()
end_time = datetime.now()
duration = end_time - start_time
print("Total time is ",duration)