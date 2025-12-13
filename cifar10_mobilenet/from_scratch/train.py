import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import timm
import copy
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("train.log"),
            logging.StreamHandler()
        ]
    )
    # Tqdm compatibility: redirect stdout/stderr? 
    # Usually standard logging + tqdm works if carefully managed, or just keep it simple.
    # We will use logging.info for messages.

setup_logging()

# ==========================================
# 1. 실험 설정 (Configuration)
# ==========================================
class Config:
    # 데이터 관련
    NUM_CLIENTS = 10
    PUBLIC_RATIO = 0.2          # 전체 데이터 중 공개 데이터 비율 (20%)
    DIRICHLET_ALPHA = 0.5       # Non-IID 정도 (작을수록 불균형 심함)
    
    # 모델 관련
    MODEL_NAME = 'mobilenetv3_small_100'
    NUM_CLASSES = 10
    IMG_SIZE = 128              # MobileNet은 32x32에서 성능이 떨어지므로 리사이징 권장
    PRETRAINED = True           # Pretrained weights 사용
    
    # 학습 관련
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32

    # Phase 1: Public Pre-training 설정
    PRETRAIN_EPOCHS = 5         # Public Pre-training 에폭
    PRETRAIN_LR = 0.001
    SKIP_PRETRAIN = False       # Public Pre-training 건너뛰기 여부
    
    # Phase 2: Federated Learning 설정
    FL_ROUNDS = 10              # 통신 라운드
    LOCAL_EPOCHS = 2            # 클라이언트 당 로컬 에폭
    LOCAL_LR = 0.01

def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation (From Scratch)")
    
    # Data Args
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--public_ratio', type=float, default=0.2, help='Public data ratio')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    
    # Model Args
    parser.add_argument('--model_name', type=str, default='mobilenetv3_small_100', help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    # Training Args
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Public pre-training epochs')
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip public pre-training')
    parser.add_argument('--fl_rounds', type=int, default=10, help='FL rounds')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs')
    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='Pre-training learning rate')
    parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate')
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    
    return parser.parse_args()

# ==========================================
# 2. 데이터 유틸리티 (Data Utils)
# ==========================================
def get_transforms():
    # CIFAR-10 통계량
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE), # MobileNet 최적화를 위한 리사이징
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_tf = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    return train_tf, test_tf

def partition_data_dirichlet(targets, num_clients, alpha):
    """Dirichlet 분포를 사용한 Non-IID 인덱스 분할"""
    min_size = 0
    N = len(targets)
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(Config.NUM_CLASSES):
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

def prepare_datasets():
    logging.info(">>> Loading & Partitioning Data...")
    train_tf, test_tf = get_transforms()
    
    # 전체 데이터셋 로드
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
    
    # 1. Public (20%) vs Private (80%) 분할
    total_len = len(full_train_dataset)
    public_len = int(total_len * Config.PUBLIC_RATIO)
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
    client_idx_map_relative = partition_data_dirichlet(private_targets, Config.NUM_CLIENTS, Config.DIRICHLET_ALPHA)
    
    # 상대 인덱스를 전체 데이터셋 기준 절대 인덱스로 변환
    client_datasets = []
    for i in range(Config.NUM_CLIENTS):
        relative_idxs = client_idx_map_relative[i]
        absolute_idxs = [private_indices[idx] for idx in relative_idxs]
        client_datasets.append(Subset(full_train_dataset, absolute_idxs))
    
    logging.info(f"Data Prepared: Public({len(public_dataset)}), Private({len(private_indices)} split to {Config.NUM_CLIENTS} clients)")
    return public_dataset, client_datasets, test_dataset

# ==========================================
# 3. 모델 유틸리티 (timm Model)
# ==========================================
def get_model():
    """timm을 사용하여 MobileNet V3 Small 로드"""
    model = timm.create_model(
        Config.MODEL_NAME, 
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES
    )

    return model.to(Config.DEVICE)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return acc, avg_loss

# ==========================================
# 4. 학습 루틴 (Training Routines)
# ==========================================
def train_centralized(model, dataset, epochs, lr, description="Training"):
    """중앙 집중식 학습 (Public Data Pre-training 용)"""
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    logging.info(f"--- {description} Start ---")
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss/(i+1)})
        # print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(loader):.4f}") # tqdm이 대체함
    return model

def train_client_local(global_weights, dataset, epochs, lr):
    """클라이언트 로컬 학습 (Federated Learning 용)"""
    # 글로벌 가중치 복사 및 로드
    model = get_model()
    model.load_state_dict(global_weights)
    model.train()
    
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # FL은 보통 SGD 사용
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model.state_dict(), len(dataset)

# ==========================================
# 5. 메인 실험 실행 (Experiment Execution)
# ==========================================
if __name__ == "__main__":
    args = get_args()
    
    # Update Config
    Config.NUM_CLIENTS = args.num_clients
    Config.PUBLIC_RATIO = args.public_ratio
    Config.DIRICHLET_ALPHA = args.alpha
    Config.IMG_SIZE = args.img_size
    Config.MODEL_NAME = args.model_name
    Config.PRETRAINED = args.pretrained
    Config.BATCH_SIZE = args.batch_size

    Config.PRETRAIN_EPOCHS = args.pretrain_epochs
    Config.FL_ROUNDS = args.fl_rounds
    Config.LOCAL_EPOCHS = args.local_epochs
    Config.PRETRAIN_LR = args.pretrain_lr
    Config.LOCAL_LR = args.local_lr
    Config.SKIP_PRETRAIN = args.skip_pretrain
    
    if args.device:
        Config.DEVICE = torch.device(args.device)
        
    logging.info(f"Updated Config: {Config.__dict__}")

    # 1. 데이터 준비
    public_data, client_datasets, test_data = prepare_datasets()
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 2. 모델 초기화
    global_model = get_model()
    
    # ====================================================
    # PHASE 1: Public Data Pre-training
    # ====================================================
    if not Config.SKIP_PRETRAIN:
        logging.info("\n>>> [Phase 1] Pre-training on Public Dataset (20%)...")
        # 사전 학습 전 성능 측정
        acc_before, _ = evaluate(global_model, test_loader)
        logging.info(f"Initial Acc (ImageNet Weights only): {acc_before:.2f}%")
        
        # 공개 데이터로 학습
        global_model = train_centralized(global_model, public_data, Config.PRETRAIN_EPOCHS, Config.PRETRAIN_LR, "Public Pre-training")
        
        acc_pre, _ = evaluate(global_model, test_loader)
        logging.info(f"Acc after Public Pre-training: {acc_pre:.2f}%")
        logging.info(">>> Phase 1 Complete. This model is now the 'Server Model'.\n")
    else:
        logging.info("\n>>> [Phase 1] Skipped. Using initial weights (ImageNet or Random) as Server Model.\n")
        # 평가 루틴을 위해 acc_pre를 측정
        acc_pre, _ = evaluate(global_model, test_loader)
        logging.info(f"Initial Acc: {acc_pre:.2f}%")
    
    # ====================================================
    # PHASE 2: Federated Learning on Private Non-IID Data
    # ====================================================
    # ====================================================
    logging.info(">>> [Phase 2] Starting Federated Learning on Private Datasets...")
    global_weights = global_model.state_dict()
    fl_accuracies = [acc_pre]
    
    # tqdm으로 Round 진행 상황 표시
    round_iterator = tqdm(range(Config.FL_ROUNDS), desc="FL Rounds")
    for round_idx in round_iterator:
        local_weights_list = []
        local_sample_counts = []
        
        # 모든 클라이언트 참여 (Simulation Full Participation)
        for client_id in range(Config.NUM_CLIENTS):
            w, count = train_client_local(
                copy.deepcopy(global_weights), 
                client_datasets[client_id], 
                Config.LOCAL_EPOCHS, 
                Config.LOCAL_LR
            )
            local_weights_list.append(w)
            local_sample_counts.append(count)
            # 진행 상황 출력 (Optional)
            # print(f"Round {round_idx+1} | Client {client_id} finished.")
            
        # FedAvg Aggregation
        total_samples = sum(local_sample_counts)
        new_weights = copy.deepcopy(global_weights)
        
        for key in new_weights.keys():
            weighted_sum = 0
            for i in range(Config.NUM_CLIENTS):
                weight_ratio = local_sample_counts[i] / total_samples
                weighted_sum += local_weights_list[i][key] * weight_ratio
            new_weights[key] = weighted_sum
            
        global_weights = new_weights
        
        # Round Evaluation
        global_model.load_state_dict(global_weights)
        round_acc, round_loss = evaluate(global_model, test_loader)
        fl_accuracies.append(round_acc)
        
        # tqdm postfix에 결과 업데이트
        round_iterator.set_postfix({'Acc': f"{round_acc:.2f}%", 'Loss': f"{round_loss:.4f}"})
        # print(f"Round {round_idx+1}/{Config.FL_ROUNDS} | Global Acc: {round_acc:.2f}% | Loss: {round_loss:.4f}")

    # ====================================================
    # 결과 시각화
    # ====================================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fl_accuracies)), fl_accuracies, marker='o', label='FL with Public Pre-training')
    plt.title(f'FL Performance (MobileNetV3, Alpha={Config.DIRICHLET_ALPHA})')
    plt.xlabel('FL Rounds (0 = After Public Pretrain)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('fl_performance.png')
    # plt.show()
    
    # ====================================================
    # 모델 저장
    # ====================================================
    save_path = "mobilenet_fl_server.pth"
    torch.save(global_model.state_dict(), save_path)
    logging.info(f"Global model saved to {save_path}")

    logging.info("\nExperiment Finished Successfully!")