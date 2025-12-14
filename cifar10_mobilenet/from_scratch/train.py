import torch
import argparse
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from models import get_model
from data_utils import prepare_datasets, visualize_client_data_distribution

# Configure logging
def setup_logging(log_file="train.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# ==========================================
# 1. 실험 설정 (Configuration)
# ==========================================
class Config:
    # 학습 관련
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32

def get_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation (From Scratch)")
    
    # Data Args
    #  - num_clients: 클라이언트 수
    #  - public_ratio: 전체 데이터 중 공개 데이터 비율 (20%)
    #  - alpha: Non-IID 정도 (작을수록 불균형 심함)
    #  - mode: NATIVE(32x32) or RESIZE(224x224)
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients')
    parser.add_argument('--public_ratio', type=float, default=0.2, help='Public data ratio')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--mode', type=str, default='NATIVE', choices=['NATIVE', 'RESIZE'], help='Image size mode: NATIVE(32x32) or RESIZE(224x224)')
    
    # Model Args
    #  - model: 모델 이름
    #  - pretrained: Pretrained weights 사용 여부
    parser.add_argument('--model', type=str, default='mobilenet', help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    # Training Args
    #  - device: Device (cuda/cpu)
    #  - batch_size: Batch size
    parser.add_argument('--device', type=str, default='', help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # Phase 1: Public Pre-training
    #  - skip_pretrain: Public Pre-training 건너뛰기 여부
    #  - pretrain_epochs: Public Pre-training 에폭
    #  - pretrain_lr: Public Pre-training 학습률
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip public pre-training')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Public pre-training epochs')
    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='Pre-training learning rate')
    # Phase 2: Federated Learning
    #  - fl_rounds: 통신 라운드
    #  - local_epochs: 클라이언트 당 로컬 에폭
    #  - local_lr: 클라이언트 당 로컬 학습률
    parser.add_argument('--fl_rounds', type=int, default=10, help='FL rounds')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs')
    parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate')
    
    # Output Control
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory to save all results (logs, plots, models)')

    return parser.parse_args()

# ==========================================
# 2. 학습 루틴 (Training Routines)
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

def train_client_local(model_name, pretrained, global_weights, dataset, epochs, lr):
    """클라이언트 로컬 학습 (Federated Learning 용)"""
    # 글로벌 가중치 복사 및 로드
    model = get_model(model_name, pretrained, num_classes=10, device=Config.DEVICE)
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
    acc = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return acc, avg_loss

def run_phase_1(args, global_model, public_data, test_loader):
    """Phase 1: Public Data Pre-training"""
    initial_weights = "ImageNet Weights" if args.pretrained else "Random Weights"
    
    if not args.skip_pretrain:
        logging.info("\n>>> [Phase 1] Pre-training on Public Dataset (20%)...")
        # 사전 학습 전 성능 측정
        acc_before, _ = evaluate(global_model, test_loader)
        logging.info(f"Initial Acc ({initial_weights}): {acc_before:.2f}%")
        
        # 공개 데이터로 학습
        global_model = train_centralized(
            global_model, 
            public_data, 
            args.pretrain_epochs, 
            args.pretrain_lr, 
            "Public Pre-training"
        )
        
        acc_pre, _ = evaluate(global_model, test_loader)
        logging.info(f"Acc after Public Pre-training: {acc_pre:.2f}%")
        logging.info(">>> Phase 1 Complete. This model is now the 'Server Model'.\n")
    else:
        logging.info(f"\n>>> [Phase 1] Skipped. Using initial weights ({initial_weights}) as Server Model.\n")
        # 평가 루틴을 위해 acc_pre를 측정
        acc_pre, _ = evaluate(global_model, test_loader)
        logging.info(f"Initial Acc: {acc_pre:.2f}%")
        
    return acc_pre, global_model

def run_phase_2(args, global_model, client_datasets, test_loader, acc_pre):
    """Phase 2: Federated Learning on Private Non-IID Data"""
    logging.info(">>> [Phase 2] Starting Federated Learning on Private Datasets...")
    global_weights = global_model.state_dict()
    fl_accuracies = [acc_pre]
    
    # tqdm으로 Round 진행 상황 표시
    round_iterator = tqdm(range(args.fl_rounds), desc="FL Rounds")
    for round_idx in round_iterator:
        local_weights_list = []
        local_sample_counts = []
        
        # 모든 클라이언트 참여 (Simulation Full Participation)
        for client_id in range(args.num_clients):
            w, count = train_client_local(
                args.model,
                args.pretrained,
                copy.deepcopy(global_weights), 
                client_datasets[client_id], 
                args.local_epochs, 
                args.local_lr
            )
            local_weights_list.append(w)
            local_sample_counts.append(count)
            # 진행 상황 출력 (Optional)
            print(f"Round {round_idx+1} | Client {client_id} finished.")
            
        # FedAvg Aggregation
        total_samples = sum(local_sample_counts)
        new_weights = copy.deepcopy(global_weights)
        
        for key in new_weights.keys():
            weighted_sum = 0
            for i in range(args.num_clients):
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
        
    return fl_accuracies, global_model

def visualize_fl_performance(fl_accuracies, alpha, save_path):
    """결과 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fl_accuracies)), fl_accuracies, marker='o', label='FL with Public Pre-training')
    plt.title(f'FL Performance (MobileNetV3, Alpha={alpha})')
    plt.xlabel('FL Rounds (0 = After Public Pretrain)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    # plt.show()

# ==========================================
# 3. 메인 실험 실행 (Experiment Execution)
# ==========================================
if __name__ == "__main__":
    args = get_args()
    
    # Config Update
    Config.BATCH_SIZE = args.batch_size
    
    if args.device:
        Config.DEVICE = torch.device(args.device)
    
    # 결과 저장 폴더 생성
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    log_path = os.path.join(args.result_dir, 'train.log')
    setup_logging(log_path)
        
    logging.info(f"Updated Config: {Config.__dict__}")

    # Determine mode: if model is ViT, force RESIZE, otherwise use args.mode
    if args.model == 'vit':
        mode = 'RESIZE'
        logging.info("Model is ViT, forcing RESIZE mode")
    else:
        mode = args.mode
        logging.info(f"Using mode: {mode}")
    
    # Determine image size based on mode
    img_size = 32 if mode == 'NATIVE' else 224
    logging.info(f"Mode: {mode} -> Image Size: {img_size}")

    # 데이터 준비
    public_data, client_datasets, test_data = prepare_datasets(
        img_size,
        args.public_ratio,
        args.num_clients,
        args.alpha,
        num_classes=10
    )
    
    # 데이터 분포 시각화
    visualize_client_data_distribution(client_datasets, num_classes=10, save_path=os.path.join(args.result_dir, 'client_distribution.png'))

    # 테스트 데이터 로더 생성
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    global_model = get_model(args.model, args.pretrained, num_classes=10, img_size=img_size, device=Config.DEVICE)
    
    # PHASE 1: Public Data Pre-training
    acc_pre, global_model = run_phase_1(args, global_model, public_data, test_loader)
    
    # PHASE 2: Federated Learning on Private Non-IID Data
    fl_accuracies, global_model = run_phase_2(args, global_model, client_datasets, test_loader, acc_pre)

    # 결과 시각화
    visualize_fl_performance(fl_accuracies, args.alpha, os.path.join(args.result_dir, 'fl_performance.png'))
    
    # 모델 저장
    save_model_path = os.path.join(args.result_dir, 'mobilenet_fl_server.pth')
    torch.save(global_model.state_dict(), save_model_path)
    logging.info(f"Global model saved to {save_model_path}")

    logging.info("\nExperiment Finished Successfully!")