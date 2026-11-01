"""Registered capacities and the commissioning posterior (newpaper papers/model_direction.md, 2.3-2.4).

A registry event is a dict with `node`, `kind` (technology index), `tau` (hour index of the
registry date, i.e. 00:00 of that day) and `delta` (kW). Synthetic data also carry `true`
(hour of the real commissioning). An event registered on day tau_d is known at forecast
origins of day tau_d + 1 and later.

The capacity of technology k that the model assumes at hour t', given what is known at the
origin, is

    cap_k(t') = cap0_k + sum over known events e of kind k of delta_e * rho_e(t')

with one of the schemes
  registry   rho = 1{t' >= tau}               the registry date is taken as the commissioning date
  prior      rho = F(t' - tau)                lag CDF, Gamma(shape, scale) in days
  posterior  rho = P(T <= day(t') | data)     single change point (CommissioningPosterior); only for
                                              technologies driven by irradiance (PV), others: prior
  oracle     rho = 1{t' >= true hour}         synthetic data only
"""
import numpy as np
from scipy import stats

SCHEMES = ("registry", "prior", "posterior", "oracle")
MIDDAY = (10, 11, 12, 13, 14, 15)
NIGHT = (0, 1, 2, 3, 4, 5)


def lag_prior(kmax=240, shape=2.0, scale=25.0):
    """Probabilities that the commissioning lag (days) is 0, 1, ..., kmax, and (last entry) that it
    exceeds kmax."""
    cdf = stats.gamma.cdf(np.arange(kmax + 2, dtype=float), a=shape, scale=scale)
    return np.append(np.diff(cdf), 1.0 - cdf[-1])


def prior_cdf_hours(dt_hours, shape=2.0, scale=25.0):
    """Lag CDF at dt_hours after the registry date (0 before it)."""
    dt = np.asarray(dt_hours, dtype=float) / 24.0
    return np.where(dt < 0, 0.0, stats.gamma.cdf(np.maximum(dt, 0.0), a=shape, scale=scale))


def daily_signature(y_scaled, irr, hod):
    """Per day: z = mean midday value - mean night value (scaled load), g = mean midday irradiance.
    y_scaled, irr, hod: (T,) with T a multiple of 24 starting at 00:00."""
    days = len(y_scaled) // 24
    y = y_scaled[: days * 24].reshape(days, 24)
    g = irr[: days * 24].reshape(days, 24)
    h = hod[:24]
    mid = np.isin(h, MIDDAY)
    night = np.isin(h, NIGHT)
    with np.errstate(invalid="ignore"):
        z = np.nanmean(y[:, mid], 1) - np.nanmean(y[:, night], 1)
        gm = np.nanmean(g[:, mid], 1)
    return z, gm


class CommissioningPosterior:
    """Posterior of the commissioning day T of one registered PV addition.

    Daily statistics z_d (midday minus night, load scaled by sigma) and g_d (mean midday
    irradiance in kW/m2). Before T: z_d = a + b g_d + e; from T on the midday load drops by the
    expected output of the new capacity, z_d = a + b g_d - eta * (delta / sigma) g_d + e, with
    e ~ N(0, s^2); a, b, s are fitted on the `ref_days` days before the registry day. Day
    log-likelihoods are multiplied by `temper` (< 1) because the residuals are correlated.
    `cdf(D, days)` = P(T <= day | data of days < D) for an origin D; days before the registry
    day get 0."""

    def __init__(self, z, g, tau_d, delta_over_sigma, prior, eta=0.8, ref_days=60, temper=0.5):
        self.tau_d, self.prior = tau_d, np.asarray(prior, dtype=float)
        self.kmax = len(self.prior) - 2
        lo = max(0, tau_d - ref_days)
        zr, gr = z[lo:tau_d], g[lo:tau_d]
        ok = np.isfinite(zr) & np.isfinite(gr)
        if ok.sum() >= 10 and np.std(gr[ok]) > 1e-3:
            b, a = np.polyfit(gr[ok], zr[ok], 1)
            res = zr[ok] - (a + b * gr[ok])
        else:                                   # too little reference data: level only
            a, b = (np.nanmean(zr) if ok.any() else 0.0), 0.0
            res = zr[ok] - a if ok.any() else np.zeros(1)
        s = max(1.4826 * np.median(np.abs(res - np.median(res))) if len(res) > 1 else 0.0, 1e-3)
        self.a, self.b, self.s = a, b, s
        post = np.arange(tau_d, len(z))
        r = z[post] - (a + b * g[post])
        drop = eta * delta_over_sigma * g[post]
        l0 = -0.5 * (r / s) ** 2
        l1 = -0.5 * ((r + drop) / s) ** 2
        bad = ~(np.isfinite(r) & np.isfinite(drop))
        l0[bad], l1[bad] = 0.0, 0.0
        self.L0 = np.concatenate([[0.0], np.cumsum(temper * l0)])     # L0[m] = sum of first m days
        self.L1 = np.concatenate([[0.0], np.cumsum(temper * l1)])
        self._cache = {}

    def weights(self, D):
        """P(T = tau_d + j), j = 0..kmax, and P(T > tau_d + kmax), given the days before D."""
        m = int(np.clip(D - self.tau_d, 0, len(self.L0) - 1))
        if m in self._cache:
            return self._cache[m]
        j = np.arange(self.kmax + 1)
        jm = np.minimum(j, m)
        logw = np.log(self.prior[:-1] + 1e-300) + self.L0[jm] + (self.L1[m] - self.L1[jm])
        logw = np.append(logw, np.log(self.prior[-1] + 1e-300) + self.L0[m])
        w = np.exp(logw - logw.max())
        w /= w.sum()
        self._cache[m] = w
        return w

    def cdf(self, D, days):
        w = self.weights(D)
        c = np.cumsum(w[:-1])
        j = np.asarray(days) - self.tau_d
        return np.where(j < 0, 0.0, c[np.clip(j, 0, self.kmax)])


def build_posteriors(events, y, cov, hod, irr_col, pv_kind, sigma_fn, prior, **kw):
    """CommissioningPosterior for every PV event (index -> posterior). sigma_fn(node, day) gives
    the load scale at the registry day."""
    out = {}
    for n, e in enumerate(events):
        if e["kind"] != pv_kind:
            continue
        i, tau_d = e["node"], e["tau"] // 24
        sig = sigma_fn(i, tau_d)
        z, g = daily_signature(y[i] / sig, cov[i, :, irr_col], hod)
        out[n] = CommissioningPosterior(z, g, tau_d, e["delta"] / sig, prior, **kw)
    return out


def capacity_window(t0, L, H, cap0, events, scheme, posteriors=None, pv_kind=None,
                    shape=2.0, scale=25.0):
    """(L + H, K) capacities assumed for hours t0 - L .. t0 + H - 1 of one node at origin t0.
    events: (index, event) pairs of this node; only events registered before the origin day are
    known."""
    hours = np.arange(t0 - L, t0 + H)
    cap = np.repeat(np.asarray(cap0, dtype=float)[None], L + H, 0)
    D = t0 // 24
    for n, e in events:
        if e["tau"] // 24 >= D:
            continue
        if scheme == "registry":
            rho = (hours >= e["tau"]).astype(float)
        elif scheme == "oracle":
            rho = (hours >= e["true"]).astype(float)
        elif scheme == "posterior" and e["kind"] == pv_kind and posteriors and n in posteriors:
            rho = posteriors[n].cdf(D, hours // 24)
        elif scheme in ("prior", "posterior"):
            rho = prior_cdf_hours(hours - e["tau"], shape, scale)
        else:
            raise ValueError(scheme)
        cap[:, e["kind"]] += e["delta"] * rho
    return cap
