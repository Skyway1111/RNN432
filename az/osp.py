"""Order-statistic gap proposals: laws on input directions built from the top three values and their gaps, with exact density.

A component draws positions (p, q, r) of the largest, 2nd and 3rd from pi, the 2nd largest S ~ N(mu, sig^2), and
nonnegative gaps: M = S + tm|z|, T = S - tt|z|, the lower input at position i T - tl_i|z_i|. The order p > q > r > lowers holds by
construction, so at a direction u whose top three sit at (p, q, r) the density along the ray is q(r u) = exp(C - A r^2 + B r),
and the angular density is closed form: int_0^inf q(r u) r^9 dr = e^C (2A)^-5 K9(B / sqrt(2A)),
K9(s) = int_0^inf t^9 exp(-t^2/2 + s t) dt. The event is scale-free, so this radial integral is exact Rao-Blackwellization.
Small tm, tt put mass on the near-tie wedges; mu < 0 with small tt, tl on inputs with at most one positive value.
"""
import math

import numpy as np
import torch as th

DEV = "cuda" if th.cuda.is_available() else "cpu"
AREA = 2 * math.pi**5 / 24  # surface of S^9; the nominal angular density is 1 / AREA
H = math.log(2 / math.sqrt(2 * math.pi))  # log of the half-normal density's constant
FLOOR = 1e-3  # smallest scale a fit may return
KEYS = 1000  # p * 100 + q * 10 + r
_S = np.linspace(-60.0, 60.0, 12001)


def _k9_table() -> th.Tensor:
    """log K9 on _S by quadrature in log t around the integrand's peak t* = (s + sqrt(s^2 + 36)) / 2."""
    t0 = (_S + np.sqrt(_S**2 + 36)) / 2
    y = np.linspace(-9.0, 4.0, 6001)
    t = t0[:, None] * np.exp(y)
    lg = 10 * np.log(t) - t**2 / 2 + _S[:, None] * t
    top = lg.max(1, keepdims=True)
    return th.tensor(top[:, 0] + np.log(np.exp(lg - top).sum(1) * (y[1] - y[0])), dtype=th.float64, device=DEV)


_K9 = _k9_table()
_DK9 = _K9[1:] - _K9[:-1]


def logk9(s: th.Tensor) -> th.Tensor:
    """log K9 by linear interpolation on [-60, 60] and the leading asymptotics outside it, evaluated only where needed."""
    x = (s.clamp(-60, 60) + 60) * ((len(_S) - 1) / 120)
    i = x.floor().long().clamp(max=len(_S) - 2)
    out = _K9[i] + (x - i) * _DK9[i]
    far = s.abs() > 60
    if far.any():
        t = s[far]
        out[far] = th.where(t < 0, math.log(math.factorial(9)) - 10 * th.log(-t),
                            t**2 / 2 + 0.5 * math.log(2 * math.pi) + 9 * th.log(t.abs()) + th.log1p(36 / t**2))
    return out


def valid_keys() -> th.Tensor:
    a = th.arange(KEYS, device=DEV)
    p, q, r = a // 100, (a // 10) % 10, a % 10
    return (p != q) & (p != r) & (q != r)



class Gaps:
    """Sorted directions and what every component's density reads from them."""

    def __init__(self, u: th.Tensor):
        v, idx = u.sort(1, descending=True)
        self.key = idx[:, 0] * 100 + idx[:, 1] * 10 + idx[:, 2]
        self.S, T = v[:, 1], v[:, 2]
        self.dm2, self.dt2 = (v[:, 0] - self.S) ** 2, (self.S - T) ** 2
        self.low = (T[:, None] - u) ** 2  # squared gap below T at every position
        self.low.scatter_(1, idx[:, :3], 0.0)


class OSP:
    """K gap-proposal components with stacked parameters: pi (K, 1000), mu, sig, tm, tt (K,), tl (K, 10), so that every
    density, placement and reorder is one vectorized op over components. tl may be a scalar or one row of 10."""

    def __init__(self, pi, mu, sig, tm, tt, tl):
        f = lambda a: th.as_tensor(a, dtype=th.float64, device=DEV)  # noqa: E731
        self.pi = f(pi).reshape(-1, KEYS)
        self.pi = self.pi / self.pi.sum(1, keepdim=True)
        K = len(self.pi)
        self.mu, self.sig, self.tm, self.tt = (f(v).reshape(-1).expand(K).clone() for v in (mu, sig, tm, tt))
        self.tl = f(tl).expand(K, 10).clone()  # lower-gap scale per component and position
        self.cdf = (self.pi.cumsum(1) + th.arange(K, device=DEV)[:, None]).reshape(-1)  # row k shifted by k: one sorted array
        a = th.arange(KEYS, device=DEV)
        top = self.tl.log()[:, th.stack([a // 100, (a // 10) % 10, a % 10], 1)].sum(2)  # the three positions not below T
        self.ckey = (th.log(self.pi.clamp(min=1e-300)) - self.tl.log().sum(1, keepdim=True) + top + 9 * H
                     - (th.log(self.sig * math.sqrt(2 * math.pi)) + th.log(self.tm * self.tt) + self.mu**2 / (2 * self.sig**2))[:, None]
                     ).T.contiguous()  # (1000, K): the constant per order, read by row

    def __len__(self):
        return len(self.mu)

    @staticmethod
    def cat(parts: list["OSP"]) -> "OSP":
        return OSP(*(th.cat([getattr(p, a) for p in parts]) for a in ("pi", "mu", "sig", "tm", "tt", "tl")))

    def params(self, k: int = 0):
        return self.pi[k].cpu().numpy(), self.mu[k].item(), self.sig[k].item(), self.tm[k].item(), self.tt[k].item(), self.tl[k].cpu().numpy()

    def sample(self, n: int, gen: th.Generator, k: int = 0) -> th.Tensor:
        """n points of component k."""
        return self.place(th.full((n,), k, device=DEV), th.rand(n, generator=gen, device=DEV, dtype=th.float64),
                          th.randn(n, 13, generator=gen, device=DEV, dtype=th.float64))

    def place(self, comp: th.Tensor, ukey: th.Tensor, z: th.Tensor) -> th.Tensor:
        """Points of components comp (N,): a uniform choosing (p, q, r) by pi's inverse CDF, and 13 standard normals z."""
        k = (th.searchsorted(self.cdf, (ukey + comp).contiguous(), right=True) - comp * KEYS).clamp(0, KEYS - 1)
        p, q, r = k // 100, (k // 10) % 10, k % 10
        S = self.mu[comp] + self.sig[comp] * z[:, 10]
        T = S - self.tt[comp] * z[:, 11].abs()
        x = T[:, None] - self.tl[comp] * z[:, :10].abs()
        i = th.arange(len(k), device=DEV)
        x[i, p], x[i, q], x[i, r] = S + self.tm[comp] * z[:, 12].abs(), S, T
        return x

    def _quad(self, g: Gaps) -> th.Tensor:
        """(N, K): A, the coefficient of -r^2 along the ray."""
        return (g.S[:, None] ** 2 / (2 * self.sig**2) + g.dm2[:, None] / (2 * self.tm**2) + g.dt2[:, None] / (2 * self.tt**2)
                + g.low @ (0.5 / self.tl**2).T)

    def log_ang(self, g: Gaps) -> th.Tensor:
        """(N, K): log angular density w.r.t. surface measure at unit directions."""
        A2 = 2 * self._quad(g)
        return self.ckey[g.key] - 5 * th.log(A2) + logk9(g.S[:, None] * (self.mu / self.sig**2) / th.sqrt(A2))

    def log_x(self, g: Gaps) -> th.Tensor:
        """(N, K): log density at points x (not directions), for EM."""
        return self.ckey[g.key] + g.S[:, None] * (self.mu / self.sig**2) - self._quad(g)

    def reordered(self, mass: th.Tensor, prior: float) -> "OSP":
        """Each component's order law from mass (K, 1000) plus prior times that mass of its own law; rows without mass keep theirs."""
        m = mass.sum(1, keepdim=True)
        pi = th.where(m > 0, (mass + prior * m * self.pi) * valid_keys(), self.pi)
        return OSP(pi, self.mu, self.sig, self.tm, self.tt, self.tl)


def fit(x: th.Tensor, w: th.Tensor, smooth: float = 0.5) -> OSP:
    """Weighted maximum likelihood of one component from points x; pi is smoothed toward uniform over the 720 orders and
    each position's lower-gap scale toward the pooled one, by `smooth` of the weight in pseudo-counts."""
    g = Gaps(x)
    W = w.sum()
    pi = (th.bincount(g.key, weights=w, minlength=KEYS) + smooth * W / 720) * valid_keys()
    mu = (w * g.S).sum() / W
    sig = ((w * (g.S - mu) ** 2).sum() / W).sqrt()
    tm, tt = ((w * g.dm2).sum() / W).sqrt(), ((w * g.dt2).sum() / W).sqrt()
    below = (g.low > 0).double()
    pooled = (w @ g.low).sum() / (w @ below).sum()
    tl = ((w @ g.low + smooth * W * pooled) / (w @ below + smooth * W)).sqrt().clamp(min=FLOOR)
    return OSP(pi, mu, *(max(v.item(), FLOOR) for v in (sig, tm, tt)), tl)


def em(x: th.Tensor, k: int, iters: int = 30, seed: int = 0):
    """Mixture of k components fitted to points x by EM from random responsibilities; returns the components and weights."""
    gen = th.Generator(device=DEV).manual_seed(seed)
    r = th.rand(len(x), k, generator=gen, device=DEV, dtype=th.float64)
    r = r / r.sum(1, keepdim=True)
    g = Gaps(x)
    for _ in range(iters):
        comps = OSP.cat([fit(x, r[:, j]) for j in range(k)])
        r = th.softmax(th.log(r.mean(0)) + comps.log_x(g), 1)
    return comps, r.mean(0)


def mixture_logs(comps: OSP, alphas: th.Tensor, beta: float, g: Gaps) -> th.Tensor:
    """(N, 1 + K): log of each part's weighted angular density, the nominal first; alphas sum to 1 - beta."""
    return th.cat([th.full((len(g.S), 1), math.log(beta / AREA), dtype=th.float64, device=DEV), th.log(alphas) + comps.log_ang(g)], 1)


def mixture_weight(comps: OSP, alphas: th.Tensor, beta: float, u: th.Tensor) -> th.Tensor:
    """Likelihood ratio nominal / (beta nominal + sum alphas comps) at directions u."""
    return th.exp(-math.log(AREA) - th.logsumexp(mixture_logs(comps, alphas, beta, Gaps(u)), 1))
