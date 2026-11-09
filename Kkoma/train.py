import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tokenizer import CharTokenizer
from model import KkomaModel, get_parameter_count

class KkomaDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            # BOS + 텍스트 + EOS
            tokens = tokenizer.encode(line, add_bos=True, add_eos=True)
            
            # max_seq_len에 맞게 자르거나 패딩 추가
            if len(tokens) > max_seq_len + 1:
                tokens = tokens[:max_seq_len + 1]
            else:
                tokens += [tokenizer.pad_id] * (max_seq_len + 1 - len(tokens))
                
            # Input(x)은 처음부터 마지막 토큰 직전까지, Target(y)은 두 번째 토큰부터 끝까지
            x = tokens[:-1]
            y = tokens[1:]
            self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train():
    # 1. 디바이스 설정 (Colab GPU 자동 인식)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. 토크나이저 및 데이터셋 로드
    tokenizer = CharTokenizer()
    tokenizer.load("vocab.json")
    
    # 하이퍼파라미터
    max_seq_len = 32
    batch_size = 16
    epochs = 1000
    learning_rate = 3e-4

    dataset = KkomaDataset("data/data.txt", tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 모델 초기화
    model = KkomaModel(
        vocab_size=tokenizer.vocab_size, 
        d_model=32, n_heads=4, n_layers=1, ffn_dim=128, max_seq_len=max_seq_len
    ).to(device)
    
    print(f"Model Parameters: {get_parameter_count(model):,}")

    # 4. 손실 함수(PAD 토큰 무시) 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    # 5. 학습 루프
    model.train()
    print("--- Kkoma 0age Training Start ---")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad() # 기울기 초기화
            logits = model(x)     # Forward (B, T, Vocab)
            
            # Loss 계산을 위해 차원 변경: (B*T, Vocab), (B*T)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            
            loss.backward()       # Backward (Autograd가 자동 계산)
            optimizer.step()      # 가중치 업데이트
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # 주기적 출력 및 Best 모델 저장
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f}")
            
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "kkoma_0age_babbling_best.pt")

    # 최종 모델 저장
    torch.save(model.state_dict(), "kkoma_0age_babbling.pt")
    print("학습 완료! 모델이 'kkoma_0age_babbling.pt'로 저장되었습니다.")

if __name__ == "__main__":
    train()