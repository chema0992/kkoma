# Kkoma 0age Babbling (PyTorch Version)

이 프로젝트는 거대한 LLM을 만들기 전, 언어 모델이 토큰을 인식하고 문맥을 파악해 다음 글자를 예측하는 **전체 파이프라인을 직접 구현하고 이해하기 위한 교육용 미니 프로젝트**입니다. 

초기 NumPy 검증 단계를 지나 PyTorch 프레임워크 기반으로 전환되었으며, Google Colab의 GPU 환경에서 쾌적하게 구동됩니다.

## 🚀 Google Colab에서 실행하기

Google Colab을 열고 코드 셀에 아래 명령어를 입력하여 순서대로 실행하세요.

```bash
# 1. 저장소 클론 (실제 저장소가 있다면 URL 변경, 없다면 로컬에서 폴더 압축 후 업로드)
git clone <YOUR_REPO_URL>
cd Kkoma

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 토크나이저 생성 (data.txt 기반으로 vocab.json 생성)
python tokenizer.py

# 4. 모델 학습 (GPU 자동 인식, 가중치 pt 파일 저장)
python train.py

# 5. 문장 생성 테스트
python generate.py