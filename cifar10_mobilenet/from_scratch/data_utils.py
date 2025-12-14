import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_transforms(img_size):
    # CIFAR-10 통계량
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    return train_tf, test_tf

def partition_data_dirichlet(targets, num_clients, alpha, num_classes):
    """Dirichlet 분포를 사용한 Non-IID 인덱스 분할"""
    min_size = 0
    N = len(targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance check (비율 보정)
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_split = np.split(idx_k, proportions)
            for i in range(num_clients):
                idx_batch[i] += idx_split[i].tolist()
        
        min_size = min([len(idx_j) for idx_j in idx_batch])
        net_dataidx_map = {i: idx_batch[i] for i in range(num_clients)}
            
    return net_dataidx_map

def prepare_datasets(img_size, public_ratio, num_clients, alpha, num_classes):
    logging.info(">>> Loading & Partitioning Data...")
    train_tf, test_tf = get_transforms(img_size)
    
    # 전체 데이터셋 로드
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    
    # 1. Public (20%) vs Private (80%) 분할
    total_len = len(full_train_dataset)
    public_len = int(total_len * public_ratio)
    private_len = total_len - public_len
    
    # 간단하게 인덱스로 분할 (앞부분 Public, 뒷부분 Private)
    indices = list(range(total_len))
    public_indices = indices[:public_len]
    private_indices = indices[public_len:]
    
    public_dataset = Subset(full_train_dataset, public_indices)
    
    # 2. Private 데이터를 10개 클라이언트에 Non-IID 분할
    # Subset은 targets 속성이 바로 없으므로 원본 데이터셋에서 타겟 추출 필요
    all_targets = np.array(full_train_dataset.targets)
    private_targets = all_targets[private_indices]
    
    # Non-IID 인덱스 생성 (Private 데이터 내부에서의 상대 인덱스)
    client_idx_map_relative = partition_data_dirichlet(private_targets, num_clients, alpha, num_classes)
    
    # 상대 인덱스를 전체 데이터셋 기준 절대 인덱스로 변환
    client_datasets = []
    for i in range(num_clients):
        relative_idxs = client_idx_map_relative[i]
        absolute_idxs = [private_indices[idx] for idx in relative_idxs]
        client_datasets.append(Subset(full_train_dataset, absolute_idxs))
    
    logging.info(f"Data Prepared: Public({len(public_dataset)}), Private({len(private_indices)} split to {num_clients} clients)")
    return public_dataset, client_datasets, test_dataset

def visualize_client_data_distribution(client_datasets, num_classes, save_path):
    """클라이언트별 데이터 분포 시각화 및 저장"""
    logging.info(">>> Visualizing Client Data Distribution...")
    client_counts = np.zeros((len(client_datasets), num_classes))
    
    for i, dataset in enumerate(client_datasets):
        # Subset의 경우 dataset.dataset.targets를 참조하고, dataset.indices를 사용해야 함
        # 하지만 여기서는 dataset이 Subset 객체이므로 순회하며 target을 얻거나, 원본 접근
        # 효율성을 위해 원본 targets에 접근
        if isinstance(dataset, Subset):
            # dataset.dataset is the Full dataset
            # dataset.indices are the indices for this client
            targets = np.array(dataset.dataset.targets)
            client_targets = targets[dataset.indices]
            
            for t in client_targets:
                client_counts[i][t] += 1
        else:
            # 일반 Dataset인 경우 (잘 없을 수 있음)
            for _, label in dataset:
                client_counts[i][label] += 1

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(client_datasets))
    bottom = np.zeros(len(client_datasets))
    
    for k in range(num_classes):
        ax.bar(x, client_counts[:, k], bottom=bottom, label=f'Class {k}')
        bottom += client_counts[:, k]
        
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Client ID')
    ax.set_title('Label Distribution per Client')
    ax.set_xticks(x)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Distribution plot saved to {save_path}")
