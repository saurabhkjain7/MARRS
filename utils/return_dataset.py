import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
from .randaugment import  RandAugmentMC


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(source, target, args, return_idx=False):
    base_path = './data/%s' % args.dataset

    if args.dataset == 'multi':
        root =  "./images/dnet"
    elif args.dataset ==  'office_home':
        root =  "./images/office_home"
    else:
        root= "./images/office"

    
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     target + '_%d.txt' % (args.shots))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     target + '_%d.txt' % (args.shots))

    crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'strong': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])}

    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,                            
                                      transform=data_transforms['train'],test=True)
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['train'],test= True)
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])

    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                              transform=data_transforms['val'], test=True)
 
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'], test= True)
    

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size= bs*2,
                                    num_workers=3,
                                    shuffle=False, drop_last=False)           
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,                      
                                    batch_size= bs*2 , num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs*2, num_workers=3,
                                    shuffle=False, drop_last=False)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list
