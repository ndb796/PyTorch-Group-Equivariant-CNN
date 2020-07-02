from torch.utils.data import Dataset

from PIL import Image
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def AIGS_Dataset(transform_train, transform_test):
    aigs10_classes = ('aeroplane', 'car', 'bird', 'cat', 'sheep', 'dog', 'chair', 'horse', 'boat', 'train')

    train_inputs = []
    train_targets = []
    for i in range(len(aigs10_classes)):
        data = np.load('./data/AIGS10/traindata/' + aigs10_classes[i] + '_train.npy')
        train_inputs += list(data)
        train_targets += list(np.full(data.shape[0], i))

    test_inputs = []
    test_targets = []
    for i in range(len(aigs10_classes)):
        data = np.load('./data/AIGS10/testdata/' + aigs10_classes[i] + '_test.npy')
        test_inputs += list(data)
        test_targets += list(np.full(data.shape[0], i))

    trainset = MyDataset(train_inputs, train_targets, transform=transform_train)
    testset = MyDataset(test_inputs, test_targets, transform=transform_test)

    return trainset, testset
