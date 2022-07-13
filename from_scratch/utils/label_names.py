def get_string(dataset, label_num):
    if dataset == 'cifar10':
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return labels[int(label_num)]
        