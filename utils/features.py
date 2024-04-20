import numpy as np
import torch

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

def get_features_worker(features_path, image_list_file_path, unique_image_list_path):
    '''
    features_path is where the neural network features are saved.
    image_list_file_path is the text file containing the list of image paths and labels.
    unique_image_list_path contains the image paths corresponding to the indices into features_path.
    '''
    to_image_paths, to_labels = make_dataset_fromlist(image_list_file_path)
    from_image_paths, from_labels = make_dataset_fromlist(unique_image_list_path)
    assert len(from_image_paths) == len(from_labels)
    assert len(from_labels) >= len(to_labels)
    assert all(np.arange(len(from_image_paths)) == from_labels) # check that I didn't make a mistake when calculating the features (make sure they are all in order)
    
    # get the index into the feature matrix
    f, y = torch.load(features_path)
    sorter = np.argsort(from_image_paths)
    ind = sorter[np.searchsorted(from_image_paths, to_image_paths, sorter=sorter)] # by this we are basically finding indices of required image paths from the unique_image_list_path
    assert all(from_image_paths[ind] == to_image_paths)
    assert all(ind == y[ind].long().numpy())
    
    out_features = f[ind]
    assert out_features.shape[0] == len(to_image_paths)
    assert out_features.shape[0] == len(to_labels)
    
    return out_features, torch.tensor(to_labels)


def get_features(augmentation, network, dataset, source, target, num):
    ''' get features '''
    source_features_path = 'feature_weights/{}_{}_{}_{}.pt'.format(augmentation, network, dataset, source)
    target_features_path = 'feature_weights/{}_{}_{}_{}.pt'.format(augmentation, network, dataset, target)
    
    source_image_list_file_path = 'data/{}/labeled_source_images_{}.txt'.format(dataset, source)
    val_target_image_list_file_path = 'data/{}/validation_target_images_{}_3.txt'.format(dataset, target)
    unlabeled_target_image_list_file_path = 'data/{}/unlabeled_target_images_{}_{}.txt'.format(dataset, target, num)
    labeled_target_image_list_file_path = 'data/{}/labeled_target_images_{}_{}.txt'.format(dataset, target, num)
    source_unique_image_list_path = 'data/{}/unique_image_paths_{}.txt'.format(dataset, source)
    target_unique_image_list_path = 'data/{}/unique_image_paths_{}.txt'.format(dataset, target)
    
    source_features, source_labels = get_features_worker(source_features_path, source_image_list_file_path, source_unique_image_list_path)
    val_target_features, val_target_labels = get_features_worker(target_features_path, val_target_image_list_file_path, target_unique_image_list_path)
    unlabeled_target_features, unlabeled_target_labels = get_features_worker(target_features_path, unlabeled_target_image_list_file_path, target_unique_image_list_path)
    labeled_target_features, labeled_target_labels = get_features_worker(target_features_path, labeled_target_image_list_file_path, target_unique_image_list_path)
    
    return source_features, source_labels, val_target_features, val_target_labels, unlabeled_target_features, unlabeled_target_labels, labeled_target_features, labeled_target_labels


def get_features_unlabeled(target,image_list, augmentation ,network ,dataset ):
    features_path = 'feature_weights/{}_{}_{}_{}.pt'.format(augmentation, network, dataset, target)
    unique_image_list_path = 'data/{}/unique_image_paths_{}.txt'.format(dataset, target)

    from_image_paths, from_labels = make_dataset_fromlist(unique_image_list_path)
    
    # get the index into the feature matrix
    f, y = torch.load(features_path)
    sorter = np.argsort(from_image_paths)
    ind = sorter[np.searchsorted(from_image_paths, image_list, sorter=sorter)] # by this we are basically finding indices of required image paths from the unique_image_list_path
    assert all(from_image_paths[ind] == image_list)
    assert all(ind == y[ind].long().numpy())
    
    out_features = f[ind]
    
    return out_features