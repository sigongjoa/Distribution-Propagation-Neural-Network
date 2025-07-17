
import torch
import torch.optim as optim
from .transformer import DistTransformer
from dpnn_lib.utils.metrics import Metrics
# from dpnn_lib.utils.inference import sample_sentence
from dpnn_lib.distributions.gaussian import GaussianDistribution

# Dummy tokenizer for demonstration
class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.id_to_token = {i: str(i) for i in range(vocab_size)}
        self.token_to_id = {str(i): i for i in range(vocab_size)}

    def encode(self, text):
        return [self.token_to_id.get(token, 0) for token in text.split()]

    def decode(self, token_ids):
        return " ".join([self.id_to_token.get(token_id, '<unk>') for token_id in token_ids])

# 1. 하이퍼파라미터 설정
vocab_size = 1000  # 예시 어휘 크기
d_model = 512      # 모델 차원
num_heads = 8      # 어텐션 헤드 수
d_ff = 2048        # 피드포워드 신경망의 내부 차원
max_len = 100      # 최대 시퀀스 길이
num_blocks = 6     # 트랜스포머 블록 수
batch_size = 32    # 배치 크기
seq_len = 50       # 시퀀스 길이

# 2. 모델 및 옵티마이저 초기화
model = DistTransformer(vocab_size, d_model, num_heads, d_ff, max_len, num_blocks)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 더미 데이터 생성
tokens = torch.randint(0, vocab_size, (batch_size, seq_len)) # (B, L)

# 4. 모델 실행 및 학습
print("Starting training...")
for epoch in range(5):
    logits, loss = model(tokens, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 결과 확인 (옵션)
    print("Logits shape:", logits.shape) # (B, L, vocab_size)
    predicted_tokens = torch.argmax(logits, dim=-1)
    print("Predicted tokens shape:", predicted_tokens.shape) # (B, L)

print("Training finished.")
