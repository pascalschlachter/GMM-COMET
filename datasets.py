import torch
import torchvision
import torchvision.transforms as T
import lightning as L
import numpy as np
import math

from networks import SourceModule


def train_transform(resize_size=256, crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((resize_size, resize_size)),
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])


def test_transform(resize_size=256, crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((resize_size, resize_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        normalize
    ])


class DropLastConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.filtered_datasets = []
        
        for dataset in datasets:
            # Calculate how many complete batches are in the dataset
            num_batches = len(dataset) // batch_size
            # Create a subset that only includes complete batches
            self.filtered_datasets.append(torch.utils.data.Subset(dataset, range(num_batches * batch_size)))
        
        # Concatenate the filtered subsets
        self.concatenated_dataset = torch.utils.data.ConcatDataset(self.filtered_datasets)
    
    def __len__(self):
        return len(self.concatenated_dataset)
    
    def __getitem__(self, idx):
        return self.concatenated_dataset[idx]


class SFUniDADataModuleBase(L.LightningDataModule):
    def __init__(self, batch_size, data_dir, category_shift, train_domain, test_domain, shared_class_num,
                 source_private_class_num, target_private_class_num):
        super(SFUniDADataModuleBase, self).__init__()
        self.batch_size = batch_size
        self.train_domain = train_domain
        self.test_domain = test_domain
        self.category_shift = category_shift

        self.batches_per_domain_accu = []

        self.train_set = None
        self.test_set = None

        self.data_dir = data_dir

        self.shared_class_num = shared_class_num
        self.source_private_class_num = source_private_class_num
        self.target_private_class_num = target_private_class_num
        self.total_class_num = shared_class_num + source_private_class_num + target_private_class_num

        self.shared_classes = [i for i in range(shared_class_num)]
        self.source_private_classes = [i + shared_class_num for i in range(source_private_class_num)]
        self.target_private_classes = [self.total_class_num - 1 - i for i in range(target_private_class_num)]

        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes

    def setup_single_test_domain(self, test_domain):
        test_set = torchvision.datasets.ImageFolder(root=self.data_dir + test_domain,
                                                         transform=test_transform())

        test_indices = [idx for idx, target in enumerate(test_set.targets) if target in self.target_classes]
        return torch.utils.data.Subset(test_set, test_indices)

    def setup(self, stage):
        # setup train set
        self.train_set = torchvision.datasets.ImageFolder(root=self.data_dir + self.train_domain,
                                                          transform=train_transform())
        train_indices = [idx for idx, target in enumerate(self.train_set.targets) if target in self.source_classes]
        self.train_set = torch.utils.data.Subset(self.train_set, train_indices)

        # setup test domain(s)
        if isinstance(self.test_domain, list):
            individual_domains = []
            for domain in self.test_domain:
                # Load dataset
                dataset = self.setup_single_test_domain(domain)
                # Shuffle dataset
                indices = np.arange(len(dataset))
                np.random.shuffle(indices)
                dataset = torch.utils.data.Subset(dataset, indices)
                individual_domains.append(dataset)
                if self.batches_per_domain_accu == []:
                    self.batches_per_domain_accu.append(math.floor(len(dataset)/self.batch_size))
                else:
                    self.batches_per_domain_accu.append(self.batches_per_domain_accu[-1] + math.floor(len(dataset)/self.batch_size))
            self.test_set = DropLastConcatDataset(individual_domains, self.batch_size)
        else:
            self.test_set = self.setup_single_test_domain(self.test_domain)

    def train_dataloader(self):
        if isinstance(self.trainer.lightning_module, SourceModule):
            return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)
        else:
            if isinstance(self.test_domain, list):
                return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                               num_workers=8)
            else:
                return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)


class Office31DataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, stage, category_shift='', train_domain='amazon', test_domain='dslr'):
        data_dir = '../../../data/public/office-31/'

        if category_shift == 'PDA':
            self.shared_class_num = 21
            self.source_private_class_num = 10
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 21
            self.source_private_class_num = 0
            self.target_private_class_num = 10
        elif category_shift == 'OPDA':
            self.shared_class_num = 10
            self.source_private_class_num = 10
            self.target_private_class_num = 11
        else:
            self.shared_class_num = 31
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        print(f"Dataset: Office-31")
        print(f"Train domain: {train_domain}")
        print(f"Test domain: {test_domain}")
        print(f"Category shift: {category_shift}")
        print(f"Shared classes: {self.shared_class_num}")
        print(f"Source privat classes: {self.source_private_class_num}")
        print(f"Target privat classes: {self.target_private_class_num}")

        super(Office31DataModule, self).__init__(batch_size, stage, data_dir, category_shift, train_domain,
                                                 test_domain, self.shared_class_num, self.source_private_class_num,
                                                 self.target_private_class_num)


class DomainNetDataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='painting', test_domain='real'):
        data_dir = '../../../../data/public/DomainNet-126/'

        if category_shift == 'PDA':
            self.shared_class_num = 200
            self.source_private_class_num = 145
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 200
            self.source_private_class_num = 0
            self.target_private_class_num = 145
        elif category_shift == 'OPDA':
            self.shared_class_num = 150
            self.source_private_class_num = 50
            self.target_private_class_num = 145
        else:
            self.shared_class_num = 345
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        if test_domain == 'continual':
            test_domain = ['clipart', 'painting', 'real', 'sketch']
            test_domain.remove(train_domain)

        super(DomainNetDataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                                  test_domain, self.shared_class_num, self.source_private_class_num,
                                                  self.target_private_class_num)


class VisDADataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='train', test_domain='test'):
        data_dir = '../../../../data/public/visda-2017/'

        if category_shift == 'OPDA':
            self.shared_class_num = 6
            self.source_private_class_num = 3
            self.target_private_class_num = 3
        elif category_shift == 'PDA':
            self.shared_class_num = 6
            self.source_private_class_num = 6
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 6
            self.source_private_class_num = 0
            self.target_private_class_num = 6
        else:
            self.shared_class_num = 12
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(VisDADataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                              test_domain, self.shared_class_num, self.source_private_class_num,
                                              self.target_private_class_num)
        

class OfficeHomeDataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, category_shift='', train_domain='Art', test_domain='Clipart'):
        data_dir = '../../../../data/public/office-home/'

        if category_shift == 'OPDA':
            self.shared_class_num = 10
            self.source_private_class_num = 5
            self.target_private_class_num = 50
        elif category_shift == 'PDA':
            self.shared_class_num = 25
            self.source_private_class_num = 40
            self.target_private_class_num = 0
        elif category_shift == 'ODA':
            self.shared_class_num = 25
            self.source_private_class_num = 0
            self.target_private_class_num = 40
        else:
            self.shared_class_num = 65
            self.source_private_class_num = 0
            self.target_private_class_num = 0

        super(OfficeHomeDataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                                   test_domain, self.shared_class_num, self.source_private_class_num,
                                                   self.target_private_class_num)
        

class CIFARDataModule(SFUniDADataModuleBase):
    def __init__(self, batch_size, data_dir, category_shift, shared_class_num, source_private_class_num, target_private_class_num, severity):
        self.severity = severity

        train_domain = None

        test_domain = ['gaussian_noise', 
                       'shot_noise', 
                       'impulse_noise',
                       'defocus_blur', 
                       'glass_blur', 
                       'motion_blur', 
                       'zoom_blur',
                       'snow', 
                       'frost', 
                       'fog', 
                       'brightness', 
                       'contrast',
                       'elastic_transform', 
                       'pixelate', 
                       'jpeg_compression']

        super(CIFARDataModule, self).__init__(batch_size, data_dir, category_shift, train_domain,
                                              test_domain, shared_class_num, source_private_class_num,
                                              target_private_class_num)

    def setup_single_test_domain(self, n_examples=10000, corruptions=["gaussian_noise"], shuffle=False):
        assert 1 <= self.severity <= 5
        n_total_cifar = 10000

        # Download labels
        labels_path = self.data_dir + 'labels.npy'
        labels = np.load(labels_path)

        x_test_list, y_test_list = [], []
        n_pert = len(corruptions)
        for corruption in corruptions:
            corruption_file_path = self.data_dir + (corruption + '.npy')

            images_all = np.load(corruption_file_path)
            images = images_all[(self.severity - 1) * n_total_cifar:self.severity *
                                n_total_cifar]
            n_img = int(np.ceil(n_examples / n_pert))
            x_test_list.append(images[:n_img])
            # Duplicate the same labels potentially multiple times
            y_test_list.append(labels[:n_img])

        x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
        if shuffle:
            rand_idx = np.random.permutation(np.arange(len(x_test)))
            x_test, y_test = x_test[rand_idx], y_test[rand_idx]

        # Make it in the PyTorch format
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        # Make it compatible with our models
        x_test = x_test.astype(np.float32) / 255
        # Make sure that we get exactly n_examples but not a few samples more
        x_test = torch.tensor(x_test)[:n_examples]
        y_test = torch.tensor(y_test)[:n_examples]

        test_set = torch.utils.data.TensorDataset(x_test, y_test)
    
        test_indices = [idx for idx, target in enumerate(test_set.tensors[1]) if target.item() in self.target_classes]
        return torch.utils.data.Subset(test_set, test_indices)
    
    def setup(self, stage):
        # setup train set
        if self.total_class_num == 10:
            self.train_set = torchvision.datasets.CIFAR10(root='../../../../data/public/', transform=T.Compose([T.ToTensor()]))
        else:
            self.train_set = torchvision.datasets.CIFAR100(root='../../../../data/public/', transform=T.Compose([T.ToTensor()]))

        train_indices = [idx for idx, target in enumerate(self.train_set.targets) if target in self.source_classes]
        self.train_set = torch.utils.data.Subset(self.train_set, train_indices)

        # setup test domain(s)
        individual_domains = []
        for domain in self.test_domain:
            # Load dataset
            dataset = self.setup_single_test_domain(corruptions=[domain])
            # Shuffle dataset
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            dataset = torch.utils.data.Subset(dataset, indices)
            individual_domains.append(dataset)
            if self.batches_per_domain_accu == []:
                self.batches_per_domain_accu.append(math.floor(len(dataset)/self.batch_size))
            else:
                self.batches_per_domain_accu.append(self.batches_per_domain_accu[-1] + math.floor(len(dataset)/self.batch_size))
        self.test_set = DropLastConcatDataset(individual_domains, self.batch_size)
        

class CIFAR100CDataModule(CIFARDataModule):
    def __init__(self, batch_size, category_shift='', severity=5):
        data_dir = '../../../../data/public/CIFAR-100-C/'

        if category_shift == 'OPDA':
            shared_class_num = 40
            source_private_class_num = 20
            target_private_class_num = 40
        elif category_shift == 'PDA':
            shared_class_num = 60
            source_private_class_num = 40
            target_private_class_num = 0
        elif category_shift == 'ODA':
            shared_class_num = 60
            source_private_class_num = 0
            target_private_class_num = 40
        else:
            shared_class_num = 100
            source_private_class_num = 0
            target_private_class_num = 0

        super(CIFAR100CDataModule, self).__init__(batch_size, data_dir, category_shift, shared_class_num, source_private_class_num,
                                                  target_private_class_num, severity)
        

class CIFAR10CDataModule(CIFARDataModule):
    def __init__(self, batch_size, category_shift='', severity=5):
        data_dir = '../../../../data/public/CIFAR-10-C/'

        if category_shift == 'OPDA':
            shared_class_num = 4
            source_private_class_num = 2
            target_private_class_num = 4
        elif category_shift == 'PDA':
            shared_class_num = 6
            source_private_class_num = 4
            target_private_class_num = 0
        elif category_shift == 'ODA':
            shared_class_num = 6
            source_private_class_num = 0
            target_private_class_num = 4
        else:
            shared_class_num = 10
            source_private_class_num = 0
            target_private_class_num = 0

        super(CIFAR10CDataModule, self).__init__(batch_size, data_dir, category_shift, shared_class_num, source_private_class_num,
                                                 target_private_class_num, severity)
