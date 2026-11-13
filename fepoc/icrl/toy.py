"""문맥 안 TD의 기억: 문맥 창 대 고정 크기 순환 기억, 그리고 플래토 게이트 쓰기. 무작위 MRP 위의 선형 가치 추정 (numpy).

배경 (Wang et al. ICLR 2025): 선형 어텐션 L층이 문맥 창의 n개 전이에 대해 배치 TD를 L번 반복한다.
    w_{l+1} = w_l + η (b_n − A_n w_l),   A_n = (1/n) Σ_j φ_j (φ_j − γ φ_{j+1})ᵀ,   b_n = (1/n) Σ_j r_j φ_j
이 계산은 문맥 전체가 아니라 (A_n, b_n)에만 의존한다. A는 같은 시각 항 +φφᵀ과 한 스텝 지연 항 −γ φ_t φ_{t+1}ᵀ의
외적 누적 (시간 비대칭 커널의 STDP 꼴 기억), b는 보상 × 상태의 외적 누적이다. 그래서 문맥 창 대신 (A, b)를 순환으로
누적하는 고정 크기 기억이면 같은 계산을 창 길이 제한 없이 할 수 있다.

memory: 흐름 길이 T에 따른 가치 오차.
    창-N (LSTD / L층)   마지막 N개 전이로 정확한 해 / L번 반복 (= 무한 깊이 / L층 선형 어텐션)
    순환 (LSTD / L층)   처음부터 누적한 (A, b)로 같은 읽기
    순환 점진           스텝마다 (A, b) 갱신 후 반복 한 번 (깊이를 시간에 나눔)
    온라인 TD(0)        상태 = w 하나, 각 전이를 한 번만 씀 (α는 T마다 가장 좋은 값)
nonstat: 보상이 단서 없이 바뀌는 흐름에서 망각 (감쇠 극점) 대 창.
gate: 플래토 게이트 쓰기. 관심 상태 i(s) ∈ {0, 1}에서만 쓰되, 게이트를 앞으로 감쇠하는 흔적으로 거른다
    F_t = i(S_t) + β F_{t−1},   쓰기 가중 = F_t        (β = 0: 생 게이트, β = γ: emphatic TD의 follow-on 흔적)
    기대 행렬 A_β = Φᵀ diag(f) (I − γP) Φ,  f = (I − β Pᵀ)⁻¹ (d_π ⊙ i).  고유값 실수부가 음이면 TD가 발산한다.
python -m fepoc.icrl.toy memory
python -m fepoc.icrl.toy gate
python -m fepoc.icrl.toy nonstat
"""
import argparse
import json
from pathlib import Path

import numpy as np


def random_mrp(rng, n=20, d=5, gamma=0.9, conc=0.3):
    """특징: 무작위 단위 벡터 (d − 1차원) + 상수 1, 전체 노름 1."""
    P = rng.dirichlet(np.full(n, conc), size=n)
    r = rng.uniform(-1, 1, n)
    Phi = rng.normal(size=(n, d - 1))
    Phi = np.concatenate([Phi / np.linalg.norm(Phi, axis=1, keepdims=True), np.ones((n, 1))], 1) / np.sqrt(2)
    ev, evec = np.linalg.eig(P.T)
    dpi = np.real(evec[:, np.argmin(np.abs(ev - 1))]); dpi = np.abs(dpi) / np.abs(dpi).sum()
    V = np.linalg.solve(np.eye(n) - gamma * P, r)
    return dict(P=P, r=r, Phi=Phi, dpi=dpi, V=V, gamma=gamma)


def msve(m, w, weight=None):
    """가중 평균제곱 가치 오차 / 가중 평균 V² (정규화)."""
    q = m["dpi"] if weight is None else m["dpi"] * weight
    err = m["Phi"] @ w - m["V"]
    return float((q * err ** 2).sum() / (q * m["V"] ** 2).sum())


def td_fixed_point(m, D=None):
    D = np.diag(m["dpi"]) if D is None else D
    A = m["Phi"].T @ D @ (np.eye(len(m["r"])) - m["gamma"] * m["P"]) @ m["Phi"]
    b = m["Phi"].T @ D @ m["r"]
    return A, b


def read(A, b, L=None, eta=0.5, ridge=1e-3):
    """L = None: 정확한 해 (A + εI)⁻¹ b. 아니면 w = 0에서 L번 반복 (L층 선형 어텐션의 TD 구성)."""
    if L is None:
        return np.linalg.solve(A + ridge * np.eye(len(b)), b)
    w = np.zeros(len(b))
    for _ in range(L):
        w = w + eta * (b - A @ w)
    return w


def stream(m, T, rng):
    n = len(m["r"])
    s = np.empty(T + 1, int)
    s[0] = rng.choice(n, p=m["dpi"])
    cum = np.cumsum(m["P"], 1)
    u = rng.random(T)
    for t in range(T):
        s[t + 1] = min(np.searchsorted(cum[s[t]], u[t]), n - 1)
    return s


def stats(m, s, lo, hi):
    """전이 lo..hi−1의 (A, b) 평균."""
    X, Y = m["Phi"][s[lo:hi]], m["Phi"][s[lo + 1:hi + 1]]
    A = X.T @ (X - m["gamma"] * Y) / (hi - lo)
    b = X.T @ m["r"][s[lo:hi]] / (hi - lo)
    return A, b


def run_memory(args):
    rng = np.random.default_rng(args.seed)
    checks = [100, 300, 1000, 3000, 10000, 30000]
    T = checks[-1]
    alphas = (0.003, 0.01, 0.03, 0.1)
    names = ["창-100 LSTD", "창-1000 LSTD", "창-1000 L=10", "순환 LSTD", "순환 L=10", "순환 점진", "온라인 TD(0)", "TD 고정점"]
    res = {k: np.zeros((args.tasks, len(checks))) for k in names}
    for task in range(args.tasks):
        m = random_mrp(rng, n=args.n, d=args.d, gamma=args.gamma)
        s = stream(m, T, rng)
        w_star = read(*td_fixed_point(m), ridge=0.0)
        # 순환 점진: (A, b) 이동 평균 + 반복 한 번 / 스텝
        Phi, gam = m["Phi"], m["gamma"]
        A_run, b_run, w_inc = np.zeros((args.d, args.d)), np.zeros(args.d), np.zeros(args.d)
        w_td = {a: np.zeros(args.d) for a in alphas}
        ci = 0
        for t in range(T):
            x, y, rr = Phi[s[t]], Phi[s[t + 1]], m["r"][s[t]]
            A_run += (np.outer(x, x - gam * y) - A_run) / (t + 1)
            b_run += (rr * x - b_run) / (t + 1)
            w_inc = w_inc + 0.5 * (b_run - A_run @ w_inc)
            for a in alphas:
                w = w_td[a]
                w += a * (rr + gam * y @ w - x @ w) * x
            if t + 1 == checks[ci]:
                Tc = t + 1
                A_all, b_all = stats(m, s, 0, Tc)
                row = {
                    "창-100 LSTD": read(*stats(m, s, max(0, Tc - 100), Tc)),
                    "창-1000 LSTD": read(*stats(m, s, max(0, Tc - 1000), Tc)),
                    "창-1000 L=10": read(*stats(m, s, max(0, Tc - 1000), Tc), L=10),
                    "순환 LSTD": read(A_all, b_all),
                    "순환 L=10": read(A_all, b_all, L=10),
                    "순환 점진": w_inc,
                    "TD 고정점": w_star,
                }
                for k, w in row.items():
                    res[k][task, ci] = msve(m, w)
                res["온라인 TD(0)"][task, ci] = min(msve(m, w_td[a]) for a in alphas)
                ci += 1
    print(f"무작위 MRP {args.tasks}개 (상태 {args.n}, 특징 {args.d}, γ {args.gamma}). 정규화 MSVE의 중앙값 [25%, 75%]")
    print("T".ljust(16) + "".join(f"{c:>20d}" for c in checks))
    for k in names:
        q = np.quantile(res[k], [0.25, 0.5, 0.75], axis=0)
        print(k.ljust(16) + "".join(f"{q[1, j]:>8.4f} [{q[0, j]:.3f},{q[2, j]:.3f}]" for j in range(len(checks))))
    d = args.d
    print(f"\n기억 크기 (실수 개수): 창-N = N·(2d+1) (N = 1000 → {1000 * (2 * d + 1)}), 순환 = d² + d = {d * d + d}")
    print(f"질의당 계산: 창-N LSTD ≈ N·d² + d³, 창 L층 ≈ N·L·d, 순환 ≈ d³ (LSTD) / L·d² (L층) / d² (점진, 스텝마다)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "memory.json").write_text(json.dumps({k: v.tolist() for k, v in res.items()} | {"checks": checks}))


def gated(m, interest, beta):
    n = len(m["r"])
    f = np.linalg.solve(np.eye(n) - beta * m["P"].T, m["dpi"] * interest)
    return td_fixed_point(m, np.diag(f))


def run_gate(args):
    """불안정: min Re λ(A) < −ε (기대 TD가 발산). 특이: |min Re λ| ≤ ε (쓴 상태가 특징 차원보다 적어 해가 하나로 안 정해짐)."""
    rng = np.random.default_rng(args.seed)
    fracs = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    eps = 1e-6
    out_all = {}
    for gamma in (0.9, 0.99):
        for kind, frac in (("무작위 20%", 0.2), ("무작위 40%", 0.4), ("보상 상위 20%", 0.2)):
            unstable = np.zeros(len(fracs)); singular = np.zeros(len(fracs)); notpd = np.zeros(len(fracs))
            err = {f: [] for f in fracs}; err_u = []
            for _ in range(args.instances):
                m = random_mrp(rng, n=args.n, d=args.d, gamma=gamma)
                k = max(1, int(round(frac * args.n)))
                idx = rng.choice(args.n, k, replace=False) if kind.startswith("무작위") else np.argsort(m["r"])[-k:]
                interest = np.zeros(args.n); interest[idx] = 1.0
                eu = msve(m, read(*td_fixed_point(m), ridge=0.0), interest)
                ok_all, es = True, {}
                for j, fr in enumerate(fracs):
                    A, b = gated(m, interest, fr * gamma)
                    lam = np.linalg.eigvals(A).real.min()
                    unstable[j] += lam < -eps
                    singular[j] += abs(lam) <= eps
                    notpd[j] += np.linalg.eigvalsh((A + A.T) / 2).min() < -eps
                    if lam < -eps:
                        ok_all = False
                    else:
                        es[fr] = msve(m, np.linalg.lstsq(A, b, rcond=None)[0], interest)
                if ok_all:
                    err_u.append(eu)
                    for fr in fracs:
                        err[fr].append(es[fr])
            key = f"γ={gamma}, 관심={kind}"
            N = args.instances
            print(f"\n{key}  (MRP {N}개, 상태 {args.n}, 특징 {args.d})")
            print("  흔적 감쇠 β/γ               " + "".join(f"{fr:>8.2f}" for fr in fracs))
            print("  불안정 (min Re λ < 0)       " + "".join(f"{u / N:>8.3f}" for u in unstable))
            print("  특이 (λ = 0, 해 불확정)     " + "".join(f"{u / N:>8.3f}" for u in singular))
            print("  비양정치 (A + Aᵀ)           " + "".join(f"{u / N:>8.3f}" for u in notpd))
            print(f"  관심 상태 오차 중앙값 (모든 β에서 발산 없음 {len(err_u)}개): 게이트 없음 {np.median(err_u):.4f} | "
                  + " ".join(f"{fr:.2f}:{np.median(err[fr]):.4f}" for fr in fracs))
            out_all[key] = dict(unstable=(unstable / N).tolist(), singular=(singular / N).tolist(),
                                notpd=(notpd / N).tolist(), err_uniform=float(np.median(err_u)),
                                err={str(f): float(np.median(err[f])) for f in fracs})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(out_all, ensure_ascii=False))


def run_nonstat(args):
    """보상 벡터가 K 스텝마다 단서 없이 r1 ↔ r2로 바뀐다. 순환 기억에 감쇠 극점 z를 두면
    A ← z A + (1 − z) φ(φ − γφ')ᵀ (b도 같음): 실효 기억 길이 1/(1 − z). 같은 실효 길이의 창과 비교한다."""
    rng = np.random.default_rng(args.seed)
    T, K, every = 20000, 2000, 100
    lens = (100, 300, 1000)
    alphas = (0.003, 0.01, 0.03, 0.1)
    names = [f"창-{N}" for N in lens] + [f"극점 1/(1−z)={N}" for N in lens] + ["순환 (망각 없음)", "온라인 TD(0)"]
    acc = {k: [] for k in names}
    split = {k: ([], []) for k in names}                    # (전환 직후 500스텝, 다음 전환 직전 1000스텝)
    for task in range(args.tasks):
        m = random_mrp(rng, n=args.n, d=args.d, gamma=args.gamma)
        r1, r2 = m["r"], rng.uniform(-1, 1, args.n)
        V1 = np.linalg.solve(np.eye(args.n) - m["gamma"] * m["P"], r1)
        V2 = np.linalg.solve(np.eye(args.n) - m["gamma"] * m["P"], r2)
        s = stream(m, T, rng)
        Phi, gam, d = m["Phi"], m["gamma"], args.d
        R = np.array([(r1 if (t // K) % 2 == 0 else r2)[s[t]] for t in range(T)])
        X, Y = Phi[s[:-1]], Phi[s[1:]]
        mem = {N: (np.zeros((d, d)), np.zeros(d)) for N in lens}
        A_all, b_all = np.zeros((d, d)), np.zeros(d)
        w_td = {a: np.zeros(d) for a in alphas}
        err = {k: [] for k in names[:-1]}; err_td = {a: [] for a in alphas}; when = []
        for t in range(T):
            x, y, rr = X[t], Y[t], R[t]
            o = np.outer(x, x - gam * y)
            for N in lens:
                z = 1 - 1 / N
                A, b = mem[N]
                mem[N] = (z * A + (1 - z) * o, z * b + (1 - z) * rr * x)
            A_all += (o - A_all) / (t + 1); b_all += (rr * x - b_all) / (t + 1)
            for a in alphas:
                w = w_td[a]
                w += a * (rr + gam * y @ w - x @ w) * x
            if (t + 1) % every == 0 and t + 1 > K:
                mm = dict(m, V=V1 if (t // K) % 2 == 0 else V2)
                for N in lens:
                    lo = max(0, t + 1 - N)
                    Aw = X[lo:t + 1].T @ (X[lo:t + 1] - gam * Y[lo:t + 1]) / (t + 1 - lo)
                    bw = X[lo:t + 1].T @ R[lo:t + 1] / (t + 1 - lo)
                    err[f"창-{N}"].append(msve(mm, read(Aw, bw)))
                    err[f"극점 1/(1−z)={N}"].append(msve(mm, read(*mem[N])))
                err["순환 (망각 없음)"].append(msve(mm, read(A_all, b_all)))
                for a in alphas:
                    err_td[a].append(msve(mm, w_td[a]))
                when.append(t % K)
        best = min(alphas, key=lambda a: np.mean(err_td[a]))
        err["온라인 TD(0)"] = err_td[best]
        when = np.array(when)
        for k in names:
            e = np.array(err[k])
            acc[k].append(e.mean())
            split[k][0].append(e[when < 500].mean()); split[k][1].append(e[when >= K - 1000].mean())
    print(f"비정상 MRP {args.tasks}개: 보상이 {K}스텝마다 단서 없이 바뀜 (상태 {args.n}, 특징 {args.d}, γ {args.gamma}).")
    print("정규화 MSVE 중앙값 (첫 전환 이후, 100스텝마다):  전체 | 전환 직후 500스텝 | 다음 전환 직전 1000스텝 | 기억 크기")
    for k in names:
        size = (int(k.split("-")[1]) * (2 * args.d + 1)) if k.startswith("창") else (args.d * args.d + args.d if "TD(0)" not in k else args.d)
        print(f"  {k:18s} {np.median(acc[k]):.4f} | {np.median(split[k][0]):.4f} | {np.median(split[k][1]):.4f} | {size}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "nonstat.json").write_text(json.dumps({k: [list(map(float, acc[k])), list(map(float, split[k][0])),
                                                      list(map(float, split[k][1]))] for k in names}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["memory", "gate", "nonstat"])
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--instances", type=int, default=2000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/icrl_toy")
    args = ap.parse_args()
    dict(memory=run_memory, gate=run_gate, nonstat=run_nonstat)[args.exp](args)


if __name__ == "__main__":
    main()
