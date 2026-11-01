"""Knowledge reliability gate (KRG; newpaper papers/model_direction.md 2.4, theory_directions A).

Statistic of a sample, from the last `hours` hours of the look-back (in units of sigma): mean,
log std and (mean at midday - mean at night) of the decomposed look-back x - u, and the profile
consistency log(mean (x - prof_R)^2) - log(mean (x - prof_short)^2), which grows when the R-day
profile no longer describes the node (the knowledge the gate protects). Each node has a
reference distribution N(mu_i, Sigma_i) fitted on its reference samples (training period; new
nodes use their own data of that period, without any training on them); Sigma_i is shrunk
towards the pooled covariance. D2 is the Mahalanobis distance and the persistence posterior is

    q = sigmoid((c - D2) / 2),  c = chi-square(4) quantile at 1 - false-alarm rate

(corollary A2: the threshold is set by the false-alarm rate, not chosen on validation data).
"""
import numpy as np
import torch
from scipy import stats

from .registry import MIDDAY, NIGHT


def window_stats(xt, hod, hours=48, x=None, pl=None, ps=None):
    """xt, hod: (B, L) -> (B, 3), or (B, 4) with the raw look-back x and the two profiles at
    the look-back hours pl, ps."""
    xd, h = xt[:, -hours:], hod[:, -hours:]
    mid = torch.isin(h, torch.tensor(MIDDAY, device=h.device)).float()
    night = torch.isin(h, torch.tensor(NIGHT, device=h.device)).float()
    sd = torch.sqrt(xd.var(1, unbiased=False) + 1e-6)
    shape = (xd * mid).sum(1) / mid.sum(1).clamp(min=1) - (xd * night).sum(1) / night.sum(1).clamp(min=1)
    out = [xd.mean(1), torch.log(sd), shape]
    if x is not None and pl is not None and ps is not None:
        xr, a, b = x[:, -hours:], pl[:, -hours:], ps[:, -hours:]
        out.append(torch.log(((xr - a) ** 2).mean(1) + 1e-4) - torch.log(((xr - b) ** 2).mean(1) + 1e-4))
    return torch.stack(out, 1)


class NodeReference:
    def __init__(self, shrink=0.3, eps=1e-4):
        self.shrink, self.eps = shrink, eps

    def fit(self, s, node):
        """s: (n, 3) numpy statistics of the reference samples, node: (n,) node indices."""
        d = s.shape[1]
        self.pool_mu = s.mean(0)
        self.pool_cov = np.cov(s.T) + self.eps * np.eye(d)
        self.mu, self.icov = {}, {}
        for i in np.unique(node):
            si = s[node == i]
            if len(si) < 10:
                mu, cov = self.pool_mu, self.pool_cov
            else:
                mu = si.mean(0)
                cov = (1 - self.shrink) * np.cov(si.T) + self.shrink * self.pool_cov
            self.mu[int(i)] = mu
            self.icov[int(i)] = np.linalg.inv(cov + self.eps * np.eye(d))
        return self

    def d2(self, s, node):
        out = np.empty(len(s))
        for i in np.unique(node):
            m = node == i
            mu = self.mu.get(int(i), self.pool_mu)
            ic = self.icov.get(int(i), np.linalg.inv(self.pool_cov))
            dd = s[m] - mu
            out[m] = np.einsum("bi,ij,bj->b", dd, ic, dd)
        return out


def chi2_threshold(p, df=4):
    return float(stats.chi2.ppf(p, df))


def posterior_q(d2, c):
    return 1.0 / (1.0 + np.exp(-(c - d2) / 2.0))
