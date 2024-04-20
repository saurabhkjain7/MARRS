import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def get_accs_and_labels(D, x_s, x_tv, x_tu, y_s, y_tv, y_tu):
    unlabeled_confidence, unlabeled_preds = torch.max(F.softmax(D(x_tu), -1), -1)
    target_acc = (torch.sum(unlabeled_preds == y_tu)/unlabeled_preds.shape[0]).item()*100

    source_confidence, source_preds = torch.max(F.softmax(D(x_s), -1), -1)
    source_acc = (torch.sum(source_preds == y_s)/source_preds.shape[0]).item()*100

    val_confidence, val_preds = torch.max(F.softmax(D(x_tv), -1), -1)
    val_acc = (torch.sum(val_preds == y_tv)/val_preds.shape[0]).item()*100

   
    return (target_acc, source_acc, val_acc, unlabeled_preds.cpu().detach(), unlabeled_confidence.cpu().detach())

def test(test_loader, net_G):
    net_G.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(test_loader)):
            inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
            outputs = net_G(inputs)

            outputs = torch.softmax(outputs, dim=1)
            max_prob, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    net_G.train()
    acc = correct / total * 100.
    return acc

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def get_unlabeled_target_labels(dataset, target, num):
    unlabeled_target_image_list_file_path = 'data/{}/unlabeled_target_images_{}_{}.txt'.format(dataset, target, num)
    _, labels = make_dataset_fromlist(unlabeled_target_image_list_file_path)
    return torch.tensor(labels).long()

def get_validation_target_labels(dataset, target, num):
    image_list_file_path = 'data/{}/validation_target_images_{}_3.txt'.format(dataset, target)
    _, labels = make_dataset_fromlist(image_list_file_path)
    return torch.tensor(labels).long()