"""점검: 시냅스 하나에 "정밀도 × 예측 오차 × 키 흔적"으로만 쓸 때, 한 에피소드 안에서 보상으로 돌아가는 길이 생기는가.

7×7 격자, 보상 칸 하나 (에피소드마다 무작위), 보상을 받으면 출발점으로 돌아간다. 가중치 학습은 없다.
키 = (칸, 행동)마다 고정된 무작위 단위벡터 (차원 d, 서로 겹친다). 기억 = 보상 예측 행 하나 w (시냅스).
  읽기:  Q(s, a) = w · k(s, a)
  오차:  δ = r + γ Q(s', a') − Q(s, a)            (γ = 0이면 바로 다음 보상만 예측)
  흔적:  e ← γλ e + k(s, a)
  쓰기:  w ← w + β δ e,   β = τ / (τ + σ²),  σ² ← σ² + η (δ² − σ²)     (정밀도: 평소 오차가 작을수록 세게)
행동: softmax(Q / 온도).

python scripts/world/trace_credit_check.py
"""
import numpy as np

N, A, T = 7, 4, 1500
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def run(seed, gamma, lam, d=128, tau=0.05, eta=0.05, temp=0.05, learn=True):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((N * N, A, d))
    K /= np.linalg.norm(K, axis=-1, keepdims=True)
    goal = (int(rng.integers(2, N)), int(rng.integers(2, N)))
    w, e, sig2 = np.zeros(d), np.zeros(d), 1.0
    pos, laps, t0 = (0, 0), [], 0

    def act(p):
        q = K[p[0] * N + p[1]] @ w
        pr = np.exp((q - q.max()) / temp)
        return int(rng.choice(A, p=pr / pr.sum())), q

    a, q = act(pos)
    for t in range(T):
        dx, dy = MOVES[a]
        nxt = (min(max(pos[0] + dx, 0), N - 1), min(max(pos[1] + dy, 0), N - 1))
        r = float(nxt == goal)
        k = K[pos[0] * N + pos[1], a]
        if r:
            nxt_pos, boot = (0, 0), 0.0
        else:
            nxt_pos = nxt
        a2, q2 = act(nxt_pos)
        if not r:
            boot = gamma * q2[a2]
        delta = r + boot - q[a]
        e = gamma * lam * e + k
        if learn:
            beta = tau / (tau + sig2)
            w = w + beta * delta * e
            sig2 += eta * (delta * delta - sig2)
        if r:
            laps.append(t + 1 - t0)
            t0, e = t + 1, np.zeros(d)
        pos, a, q = nxt_pos, a2, q2
    return laps, abs(goal[0]) + abs(goal[1])


def summarize(name, **kw):
    first, later, n_laps, opt = [], [], [], []
    for s in range(200):
        laps, dist = run(s, **kw)
        n_laps.append(len(laps)); opt.append(dist)
        if laps:
            first.append(laps[0])
        if len(laps) >= 6:
            later.append(np.mean(laps[5:]))
    print(f"{name:34s} 보상 횟수 {np.mean(n_laps):6.1f} | 첫 도달 {np.mean(first):6.1f}스텝 | 6번째 이후 {np.mean(later) if later else float('nan'):6.1f}스텝"
          f" | 최단 {np.mean(opt):.1f}스텝")


summarize("쓰기 없음 (무작위 걸음)", gamma=0.9, lam=0.8, learn=False)
summarize("바로 다음 보상만 예측 (γ=0)", gamma=0.0, lam=0.0)
summarize("앞으로의 보상 예측, 흔적 없음 (λ=0)", gamma=0.9, lam=0.0)
summarize("앞으로의 보상 예측 + 키 흔적", gamma=0.9, lam=0.8)
summarize("같음, 키 차원 32 (키 196개)", gamma=0.9, lam=0.8, d=32)
