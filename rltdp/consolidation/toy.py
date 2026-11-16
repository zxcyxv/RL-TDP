# 빠른 시냅스 (칼만, 용량 d) + 느린 저장소 (선형, SGD, 용량 D)의 공고화 토이. numpy만 씀.
#
# 항목 i: 빠른 주소 z_i ∈ R^d (단위, 해마 색인), 느린 특징 φ_i ∈ R^D (단위), 값 y_i ∈ R^m.
# 흐름: A 항목 NA개를 reps번씩 섞어 제시 → B 항목 NB개를 reps번씩. 시간 = 제시 횟수.
#
# 빠른 기억 (정보형, 곱하는 망각, 사전은 감쇠하지 않음; HANDOVER 0.2절과 같은 꼴):
#   H ← λH + z zᵀ/R,  G ← λG + t zᵀ/R,  M = G (H + I/p0)⁻¹,  읽기 M z.
# 느린 저장소: S ← S + η (목표 − Sφ) φᵀ (제시마다 한 걸음, "느림" = 작은 η).
#
# 변형 (빠른 기억이 무엇을 담나 / 느린 쪽이 무엇으로 배우나):
#   fast      : 느린 쪽 없음. t = y.
#   slow      : 빠른 쪽 없음. 느린 쪽이 y로 배움.
#   naive     : 느린 쪽은 자기 오차 (y − Sφ)로, 빠른 쪽은 쓰는 순간의 잔차 t = y − S_쓸때 φ (그 뒤 S가 바뀌어도 그대로).
#   cr        : "현재 느린 모델에 대한 잔차" (consistent residual). 교차 통계 C ← λC + φ zᵀ/R를 같이 들고
#               G = G_y − S C. S가 어떻게 바뀌든 빠른 기억은 언제나 y − S_지금 φ의 사후 평균이다 (누적합이라 병렬 가능).
#   cr_total  : cr의 빠른 쪽 + 느린 쪽이 전체 오차 (y − Sφ − Mz)로 배움 (두 상태 운동 적응 모형의 꼴).
#   *+replay  : K번 제시마다 "잠": 본 항목 전부의 입력 (φ, z)를 다시 꺼내, 목표 = 지금의 믿음 (Sφ + Mz)으로
#               느린 쪽을 n_pass번 훑음. naive는 빠른 쪽이 그대로라 이중 계산, cr은 빠른 쪽이 자동으로 줄어든다.
#   raw+replay: DQN식. 관측 (φ, y) 전부를 버퍼에 두고 그 y로 느린 쪽을 훑음 (빠른 쪽은 cr).
#
# --change: A 항목의 절반은 첫 제시 뒤 값이 바뀜 (평가는 마지막 값 기준). 날것 재생은 옛 값도 되풀이한다.
import argparse, json
import numpy as np


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make(rng, T, NA, NB, d, D, m, reps, change):
    N = NA + NB
    z = unit(rng.standard_normal((T, N, d)))
    phi = unit(rng.standard_normal((T, N, D)))
    y = rng.standard_normal((T, N, m)) / np.sqrt(m)
    y2 = y.copy()
    chg = np.zeros(N, bool)
    if change:
        chg[: NA // 2] = True
        y2[:, chg] = rng.standard_normal((T, chg.sum(), m)) / np.sqrt(m)
    order = []
    for lo, hi in [(0, NA), (NA, N)]:
        s = np.repeat(np.arange(lo, hi), reps)
        rng.shuffle(s)
        order.append(s)
    return z, phi, y, y2, chg, np.concatenate(order)


class Fast:
    def __init__(self, T, d, D, m, lam, p0, R):
        self.lam, self.R, self.p0 = lam, R, p0
        self.H = np.zeros((T, d, d))
        self.Gy = np.zeros((T, m, d))
        self.C = np.zeros((T, D, d))
        self.Gr = np.zeros((T, m, d))
        self.d = d

    def write(self, z, phi, y, t_naive):
        l, R = self.lam, self.R
        self.H = l * self.H + np.einsum('ti,tj->tij', z, z) / R
        self.Gy = l * self.Gy + np.einsum('ti,tj->tij', y, z) / R
        self.C = l * self.C + np.einsum('ti,tj->tij', phi, z) / R
        self.Gr = l * self.Gr + np.einsum('ti,tj->tij', t_naive, z) / R

    def M(self, S, mode):
        G = self.Gr if mode == 'naive' else (self.Gy if S is None else self.Gy - S @ self.C)
        A = self.H + np.eye(self.d) / self.p0
        return np.linalg.solve(A, np.swapaxes(G, 1, 2)).swapaxes(1, 2)   # G A⁻¹ (A 대칭)


def run(variant, args, seed=0):
    rng = np.random.default_rng(seed)
    T, NA, NB, d, D, m = args.T, args.NA, args.NB, args.d, args.D, args.m
    z, phi, y, y2, chg, order = make(rng, T, NA, NB, d, D, m, args.reps, args.change)
    base = variant.split('+')[0]
    replay = variant.endswith('+replay')
    use_fast = base != 'slow'
    use_slow = base != 'fast'
    mode = 'naive' if base == 'naive' else 'cr'
    F = Fast(T, d, D, m, args.lam, args.p0, args.R)
    S = np.zeros((T, m, D)) if use_slow else None
    seen_count = np.zeros(NA + NB, int)
    buf = []                                    # raw 재생용 (항목, 그때 관측한 값)
    tr = np.arange(T)
    rec = {}

    def read(idx, Mm):
        p = np.zeros((T, len(idx), m))
        if use_slow:
            p += np.einsum('tmd,tnd->tnm', S, phi[:, idx])
        if use_fast:
            p += np.einsum('tmd,tnd->tnm', Mm, z[:, idx])
        return p

    def evaluate(tag):
        Mm = F.M(S, mode) if use_fast else None
        tgt = np.where(chg[None, :, None], y2, y)
        for g, idx in [('A', np.arange(NA)), ('B', np.arange(NA, NA + NB))]:
            p = read(idx, Mm)
            rec[f'{tag}_{g}'] = float(((p - tgt[:, idx]) ** 2).sum(-1).mean() / (tgt[:, idx] ** 2).sum(-1).mean())
            if use_slow:
                ps = np.einsum('tmd,tnd->tnm', S, phi[:, idx])
                rec[f'{tag}_slow_{g}'] = float(((ps - tgt[:, idx]) ** 2).sum(-1).mean() / (tgt[:, idx] ** 2).sum(-1).mean())
            if use_fast:
                pf = np.einsum('tmd,tnd->tnm', Mm, z[:, idx])
                rec[f'{tag}_fastnorm_{g}'] = float((pf ** 2).sum(-1).mean() / (tgt[:, idx] ** 2).sum(-1).mean())

    def sleep():
        seen = np.flatnonzero(seen_count > 0)
        if base == 'raw':
            items = np.array([b[0] for b in buf])
            vals = np.stack([b[1] for b in buf], 1)         # (T, n_obs, m)
        else:
            Mm = F.M(S, mode) if use_fast else None
            items = seen
            vals = read(seen, Mm)                            # 지금의 믿음
        for _ in range(args.n_pass):
            for j in rng.permutation(len(items)):
                f = phi[:, items[j]]
                e = vals[:, j] - np.einsum('tmd,td->tm', S, f)
                S[...] += args.eta * np.einsum('tm,td->tmd', e, f)

    for step, i in enumerate(order):
        yi = y2[:, i] if (chg[i] and seen_count[i] >= 1) else y[:, i]
        Mm = F.M(S, mode) if (use_fast and base == 'cr_total') else None
        if use_slow:
            pred_s = np.einsum('tmd,td->tm', S, phi[:, i])
            if base == 'cr_total':
                err = yi - pred_s - np.einsum('tmd,td->tm', Mm, z[:, i])
            else:
                err = yi - pred_s
            t_naive = yi - pred_s
        else:
            t_naive = yi
        if use_fast:
            F.write(z[:, i], phi[:, i], yi, t_naive)
        if use_slow:
            S += args.eta * np.einsum('tm,td->tmd', err, phi[:, i])
        seen_count[i] += 1
        buf.append((i, yi.copy()))
        if replay and (step + 1) % args.K == 0:
            sleep()
        if step + 1 == args.NA * args.reps:
            evaluate('endA')
    evaluate('endB')
    return rec


VARIANTS = ['fast', 'slow', 'naive', 'cr', 'cr_total', 'naive+replay', 'cr+replay', 'raw+replay']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=64)        # 독립 시행 (항목 배치) 수
    ap.add_argument('--NA', type=int, default=32)
    ap.add_argument('--NB', type=int, default=32)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--D', type=int, default=128)
    ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--lam', type=float, default=0.98)
    ap.add_argument('--p0', type=float, default=1.0)
    ap.add_argument('--R', type=float, default=0.01)
    ap.add_argument('--eta', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--n_pass', type=int, default=5)
    ap.add_argument('--change', action='store_true')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = {}
    for v in a.variants.split(','):
        res[v] = run(v, a)
        r = res[v]
        print(f"{v:13s} endA_A {r['endA_A']:.3f} | endB_A {r['endB_A']:.3f} endB_B {r['endB_B']:.3f}"
              + (f" | slow_A {r['endB_slow_A']:.3f} slow_B {r['endB_slow_B']:.3f}" if 'endB_slow_A' in r else '')
              + (f" | fastnorm_B {r['endB_fastnorm_B']:.3f}" if 'endB_fastnorm_B' in r else ''), flush=True)
    if a.out:
        json.dump(dict(args=vars(a), results=res), open(a.out, 'w'), indent=1)
