import json
import os

class CharTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        self.chars = []
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        # 텍스트에서 고유 문자 추출 및 정렬
        unique_chars = sorted(list(set(text.replace("\n", ""))))
        self.chars = self.special_tokens + unique_chars
        self.vocab_size = len(self.chars)
        
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text, add_bos=False, add_eos=False):
        """문자열을 토큰 ID 리스트로 변환"""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
            
        for ch in text:
            ids.append(self.char2idx.get(ch, self.unk_id))
            
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """토큰 ID 리스트를 문자열로 변환"""
        text = []
        for i in ids:
            if skip_special_tokens and i in [self.pad_id, self.unk_id, self.bos_id, self.eos_id]:
                continue
            text.append(self.idx2char.get(i, "<UNK>"))
        return "".join(text)

    def save(self, filepath="vocab.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False, indent=2)

    def load(self, filepath="vocab.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            self.chars = json.load(f)
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

if __name__ == "__main__":
    if not os.path.exists("data/data.txt"):
        print("data/data.txt 파일이 없습니다.")
        exit()
        
    with open("data/data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save("vocab.json")
    print(f"단어장 생성 완료! 총 Vocab Size: {tokenizer.vocab_size}")