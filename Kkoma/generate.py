import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from model import KkomaModel
import os

def generate_text(model, tokenizer, prompt, device, max_new_tokens=30, temperature=0.8):
    model.eval() # 평가 모드 전환
    
    # 프롬프트를 ID로 변환 (시작 시 BOS 포함)
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\n[Prompt]: {prompt}")
    print("[Output]: ", end="", flush=True)
    
    generated_ids = []
    
    with torch.no_grad(): # 추론 시 기울기 계산 비활성화
        for _ in range(max_new_tokens):
            # 컨텍스트가 모델 최대 길이를 넘지 않도록 자르기
            x_cond = x if x.size(1) <= model.max_seq_len else x[:, -model.max_seq_len:]
            
            logits = model(x_cond)
            # 마지막 시점의 토큰 확률만 추출
            next_token_logits = logits[0, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # EOS(문장 끝) 토큰이 나오면 생성 종료
            if next_token_id == tokenizer.eos_id:
                break
                
            generated_ids.append(next_token_id)
            
            # 생성된 토큰을 실시간 출력
            char = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(char, end="", flush=True)
            
            # 다음 입력을 위해 생성된 토큰을 뒤에 이어 붙이기
            x = torch.cat((x, torch.tensor([[next_token_id]], device=device)), dim=1)
            
    print("\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = CharTokenizer()
    if not os.path.exists("vocab.json") or not os.path.exists("kkoma_0age_babbling.pt"):
        print("단어장이나 학습된 모델 파일이 없습니다. train.py를 먼저 실행하세요.")
        exit()
        
    tokenizer.load("vocab.json")
    
    # train.py와 완벽히 동일한 구조로 모델 초기화
    model = KkomaModel(
        vocab_size=tokenizer.vocab_size, 
        d_model=32, n_heads=4, n_layers=1, ffn_dim=128, max_seq_len=32
    ).to(device)
    
    # 학습된 가중치 불러오기
    model.load_state_dict(torch.load("kkoma_0age_babbling.pt", map_location=device, weights_only=True))
    
    prompts = ["나는", "컴퓨터는", "Web"]
    for p in prompts:
        generate_text(model, tokenizer, prompt=p, device=device, temperature=0.5)