from collections import defaultdict
import pickle
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet import ResNet18, SupConResNet18, ResNet34, SupConResNet34, ResNet50, SupConResNet50

from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler
import random

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0
class ImageHelper(Helper):


    def poison(self):
        return

    def create_model(self):
        if self.params['dataset']=='cifar10':
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet18(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet18(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet34(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet34(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet50':
                local_model = ResNet50(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet50(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet50(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet50(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()

        elif self.params['dataset']=='cifar100':
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet18(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet34(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet50':
                local_model = ResNet50(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet50(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet50(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet50(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
        
        elif self.params['dataset']=='emnist':
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                contrastive_model = SupConResNet18(name='Contrastive',
                                created_time=self.params['current_time'], dataset='emnist')
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                contrastive_model = SupConResNet34(name='Contrastive',
                                created_time=self.params['current_time'], dataset='emnist')
                replace_model.cuda()
                contrastive_model.cuda()


        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        self.replace_model = replace_model
        self.contrastive_model = contrastive_model

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                if self.params['semantic_backdoor']:
                    continue
            
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
   
    def sample_dirichlet_train_data_without_8(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 8:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[1])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n!=8:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def sample_dirichlet_train_data_without_2(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 2:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n != 2:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def sample_dirichlet_train_data_without_1(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 1:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n != 1:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def add_trigger(self, data, pattern_diffusion=0):
        new_data = np.copy(data)
        channels, height, width = new_data.shape
        if self.params['pattern_type'] == 1:
            for c in range(channels):
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100':
                    new_data[c, height-3, width-3] = 255
                    new_data[c, height-2, width-4] = 255
                    new_data[c, height-4, width-2] = 255
                    new_data[c, height-2, width-2] = 255
                elif self.params['dataset'] == 'emnist':
                    new_data[c, height-3, width-3] = 1
                    new_data[c, height-2, width-4] = 1
                    new_data[c, height-4, width-2] = 1
                    new_data[c, height-2, width-2] = 1
        
        elif self.params['pattern_type'] == 2:
            change_range = 4
            
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100':
                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(6+diffusion)] = 255
                new_data[1, height-(6+diffusion), width-(6+diffusion)] = 255
                new_data[2, height-(6+diffusion), width-(6+diffusion)] = 255

                diffusion = 0
                new_data[0, height-(5+diffusion), width-(5+diffusion)] = 255
                new_data[1, height-(5+diffusion), width-(5+diffusion)] = 255
                new_data[2, height-(5+diffusion), width-(5+diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(6+diffusion)] = 255
                new_data[1, height-(4-diffusion), width-(6+diffusion)] = 255
                new_data[2, height-(4-diffusion), width-(6+diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(4-diffusion)] = 255
                new_data[1, height-(6+diffusion), width-(4-diffusion)] = 255
                new_data[2, height-(6+diffusion), width-(4-diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(4-diffusion)] = 255
                new_data[1, height-(4-diffusion), width-(4-diffusion)] = 255
                new_data[2, height-(4-diffusion), width-(4-diffusion)] = 255
            elif self.params['dataset'] == 'emnist':
                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(6+diffusion)] = 1

                diffusion = 0
                new_data[0, height-(5+diffusion), width-(5+diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(6+diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(4-diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(4-diffusion)] = 1
        return torch.Tensor(new_data)

    def label_dataset(self):
        label_inter_dataset_list = []
        label_fac_dataset_list = []
        pos = self.params['poison_images'][0]
        _,label_pos = self.train_dataset[pos]
        if self.params['edge_case'] and self.params['dataset']=='cifar10':
            self.inter_label = 0
        else:
            self.inter_label = label_pos

        self.fac_label = self.params['poison_label_swap']

        for ind,x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            elif label == label_pos:
                label_inter_dataset_list.append(x)
            elif label == self.params['poison_label_swap']:
                label_fac_dataset_list.append(x)

        
        return label_inter_dataset_list, label_fac_dataset_list

    def poison_dataset(self):
        indices = list()
        range_no_id = list(range(50000))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id and self.params['semantic_backdoor']:
                range_no_id.remove(image)

        #if self.params['regularize_batch']:
        #    for ind,x in enumerate(self.train_dataset):
        #        _,label = x
        #        if ind not in self.params['poison_images'] + self.params['poison_images_test']:
        #            if label == 1 or label == 2:
        #                range_no_id.remove(ind)
        
        # add random images to other parts of the batch
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.params['batch_size'])
            # range_iter[0] = self.params['poison_images'][0]
            indices.extend(range_iter)
            # range_iter = random.sample(range_no_id,
            #            self.params['batch_size']
            #                -len(self.params['poison_images'])*self.params['poisoning_per_batch'])
            # for i in range(0, self.params['poisoning_per_batch']):
            #     indices.extend(self.params['poison_images'])
            # indices.extend(range_iter)

        ## poison dataset size 64 \times 200 (64: batch size, 200 batch)
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def poison_dataset_label_5(self):
        indices = list()
        poison_indices = list()
        for ind,x in enumerate(self.test_dataset):
            _,label = x
            if label == 5:
                poison_indices.append(ind)
        
        while len(indices)<self.params['size_of_secret_dataset_label_flip']:
            range_iter = random.sample(poison_indices,np.min([self.params['batch_size'], len(poison_indices) ]))
            indices.extend(range_iter)

        self.poison_images_ind = indices

        ## poison dataset size 64 \times 200 (64: batch size, 200 batch)
        return torch.utils.data.DataLoader(self.test_dataset,
                               batch_size=self.params['batch_size'],
                               sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind))

    def poison_test_dataset_label_5(self):

        return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=self.params['test_batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                self.poison_images_ind
                            ))


    def poison_test_dataset_with_9(self):
        indices = list()
        count = 0
        for ind, x in enumerate(self.train_dataset):
            _, label =  x
            if label == 9:
                count += 1
                indices.append(ind)
            if count == 1000:
                break
        
        return torch.utils.data.DataLoader(self.train_dataset,
                            batch_size=self.params['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        

    def poison_test_dataset(self):
        #
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range(1000)
                           ))
    
    def get_test_without_label_9(self):
        indices = list()
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label == 9:
                continue
            indices.append(ind)
        #logger.info(f'test_dataset_without_label_9 : {indices}')
        return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=self.params['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def load_edge_case(self):
        with open('./data/edge-case/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
        with open('./data/edge-case/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)        

        return saved_southwest_dataset_train, saved_southwest_dataset_test

    def load_data(self):
        logger.info('Loading data')

        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_train_poison = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_train_grad = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        

        transform_test = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        self.transform_poison = transform_train_poison
        self.transform_test = transform_test

        self.train_dataset_poison = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=None)
        if self.params['dataset'] == 'cifar10':
            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transform_train)

            self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        elif self.params['dataset'] == 'cifar100':
            self.train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                            transform=transform_train)

            self.test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
        elif self.params['dataset'] == 'emnist':
            self.train_dataset = datasets.EMNIST('./data', train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            self.test_dataset = datasets.EMNIST('./data', train=False, split="mnist", transform=transform_emnist)
        
        self.train_dataset_grad = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transform_train_grad)
                                         
        ## dirichlet sampling for every participant
        ## 50000 images to 100 participant, 500 images per paricipant on average
        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            
            indices_per_participant_without_8 = self.sample_dirichlet_train_data_without_8(
                    self.params['number_of_total_participants'],
                    alpha=self.params['dirichlet_alpha'])

            indices_per_participant_without_1 = self.sample_dirichlet_train_data_without_1(
                    self.params['number_of_total_participants'],
                    alpha=self.params['dirichlet_alpha'])

            indices_per_participant_without_2 = self.sample_dirichlet_train_data_without_2(
                    self.params['number_of_total_participants'],
                    alpha=self.params['dirichlet_alpha'])
            
            indices_per_participant = self.sample_dirichlet_train_data(
                    self.params['number_of_total_participants'],
                    alpha=self.params['dirichlet_alpha'])
            
            train_loaders_without_8 = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant_without_8.items()]
            train_loaders_without_1 = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant_without_1.items()]
            train_loaders_without_2 = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant_without_2.items()]
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
            benign_train_data = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            # splitted to 500 images per participant
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]
        
        self.benign_train_data = benign_train_data
        self.train_data = train_loaders
        self.train_data_without_1 = train_loaders_without_1
        self.train_data_without_2 = train_loaders_without_2
        self.train_data_without_8 = train_loaders_without_8
       
        self.edge_poison_train, self.edge_poison_test = self.load_edge_case()
        self.label_5_poison_dataset = self.poison_dataset_label_5()
        self.label_5_test_dataset = self.poison_test_dataset_label_5()
       
        #self.poisoned_data_for_contrastive = self.poison_dataset_contrastive()
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()
        self.label_inter_dataset, self.label_fac_dataset = self.label_dataset()
        self.poison_test_data_with_9 = self.poison_test_dataset_with_9()
        self.test_data_without_9 = self.get_test_without_label_9()
        # self.params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
        #                            random.sample(range(len(train_loaders)),
        #                                          self.params['number_of_adversaries'] - 1)
        # logger.info(f"Poisoned following participants: {self.params['adversary_list']}")


    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader


    def get_secret_loader(self):
        """
        For poisoning we can use a larger data set. I don't sample randomly, though.

        """
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target