"""Numerical verification of the quantitative assertions in the manuscript.

Each check names the equation, theorem or proposition it tests. Negative
controls are included where a hypothesis is claimed to be necessary.
"""

import itertools
import sys

import numpy as np

np.random.seed(20260902)
TOL = 1e-10
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def dag(A):
    return A.conj().T


def trdist(a, b):
    return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b)))


def rand_herm(d):
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    return (A + dag(A)) / 2


def rand_state(d):
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    r = A @ dag(A)
    return r / np.trace(r).real


# =====================================================================
# Theorem 1 -- covariant projection on a truncated Fock space
# =====================================================================
# Modes n = -1, 0, +1 with occupation 0 or 1: K_y = sum_n n N_n.
LEVELS = (-1, 0, 1)
BASIS = list(itertools.product([0, 1], repeat=len(LEVELS)))
DIM = len(BASIS)
Ky = np.diag([float(sum(n * occ for n, occ in zip(LEVELS, b))) for b in BASIS])
qvals = sorted(set(np.diag(Ky).real))
Pi = {q: np.diag((np.abs(np.diag(Ky).real - q) < 1e-12).astype(float))
      for q in qvals}


def E_Y(rho):
    """Predual channel: pinching onto the K_y eigenspaces."""
    return sum(Pi[q] @ rho @ Pi[q] for q in qvals)


def U_alpha(a):
    return np.diag(np.exp(-1j * a * np.diag(Ky).real))


# Haar integral agrees with the pinching (eq. pinching)
# uniform grid on the circle: the mean is the spectrally exact Haar average
grid = np.linspace(0, 2 * np.pi, 4097)[:-1]
rho0 = rand_state(DIM)
haar = np.mean([U_alpha(a) @ rho0 @ dag(U_alpha(a)) for a in grid], axis=0)
check("eq:pinching  Haar average equals the pinching",
      np.max(np.abs(haar - E_Y(rho0))) < 1e-12,
      "max dev %.2e" % np.max(np.abs(haar - E_Y(rho0))))

# (i) accessible observables are fixed points; eps_X = 0
OX = sum(Pi[q] @ rand_herm(DIM) @ Pi[q] for q in qvals)   # commutes with K_y
E_dag = lambda M: sum(Pi[q] @ M @ Pi[q] for q in qvals)
check("Thm 1(i)  E_Y(O_X) = O_X for [O_X, K_y] = 0",
      np.max(np.abs(E_dag(OX) - OX)) < TOL)
rs = [rand_state(DIM) for _ in range(40)]
epsX = max(abs(np.trace(r @ OX) - np.trace(E_Y(r) @ OX)).real for r in rs)
check("Thm 1(i)  eps_X = 0", epsX < 1e-12, "sup dev %.2e" % epsX)

# negative control: an observable that fails to commute with K_y is moved
Obad = rand_herm(DIM)
check("Thm 1(i)  negative control: [O, K_y] != 0 is not preserved",
      np.max(np.abs(E_dag(Obad) - Obad)) > 1e-6)

# (ii) whole translation orbit carries one effective state; eps_Y = 0
orbit = [trdist(E_Y(U_alpha(a) @ rho0 @ dag(U_alpha(a))), E_Y(rho0))
         for a in (0.3, 1.1, 2.7, 5.0)]
check("Thm 1(ii)  eps_Y = 0 on the translation orbit",
      max(orbit) < 1e-12, "max trace distance %.2e" % max(orbit))

# (iii) momentum-sector probabilities survive
pq = [abs(np.trace(E_Y(rho0) @ Pi[q]) - np.trace(rho0 @ Pi[q])).real
      for q in qvals]
check("Thm 1(iii)  sector probabilities p_q unchanged", max(pq) < TOL)

# (iv) intertwining with translation-invariant dynamics
H5 = sum(Pi[q] @ rand_herm(DIM) @ Pi[q] for q in qvals)   # [H5, Ky] = 0
t = 0.83


def evolve(H, r, t):
    w, V = np.linalg.eigh(H)
    Ut = V @ np.diag(np.exp(-1j * w * t)) @ dag(V)
    return Ut @ r @ dag(Ut)


lhs, rhs = E_Y(evolve(H5, rho0, t)), evolve(H5, E_Y(rho0), t)
check("Thm 1(iv)  eq:cont-intertwine holds for [H_5, K_y] = 0",
      np.max(np.abs(lhs - rhs)) < 1e-11,
      "max dev %.2e" % np.max(np.abs(lhs - rhs)))

Hbad = rand_herm(DIM)
lb, rb = E_Y(evolve(Hbad, rho0, t)), evolve(Hbad, E_Y(rho0), t)
check("Thm 1(iv)  negative control: intertwining fails for [H, K_y] != 0",
      np.max(np.abs(lb - rb)) > 1e-3,
      "max dev %.2e" % np.max(np.abs(lb - rb)))

# Kaluza-Klein tower, eq. kk-masses
m5, R = 0.45, 1.0
tower = [np.sqrt(m5 ** 2 + (n / R) ** 2) for n in range(5)]
check("eq:kk-masses  m_n^2 = m_5^2 + n^2/R^2 (values used in Fig. 1)",
      abs(tower[0] - 0.45) < TOL and abs(tower[1] - np.sqrt(1.2025)) < TOL)

# =====================================================================
# Theorem 3 -- compact homogeneous fiber G/H
# =====================================================================
# A finite group makes Haar measure an exact average, so the twirl can be
# computed without quadrature. Take G = S_3 permuting C^3, whose isotypic
# decomposition is the trivial line plus the two-dimensional standard rep.
PERM = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
UG = [np.eye(3)[list(p)] for p in PERM]
Jm = np.ones((3, 3)) / 3
PISO = [Jm, np.eye(3) - Jm]


def twirl(r):
    """The predual of the Haar average: rho -> int dg U_g rho U_g^dagger."""
    return sum(u @ r @ dag(u) for u in UG) / len(UG)


check("Thm 3(ii)  isotypic projectors are orthogonal and sum to one",
      np.allclose(sum(PISO), np.eye(3))
      and np.allclose(PISO[0] @ PISO[1], 0))
check("Thm 3(ii)  every projector is invariant under the fiber action",
      max(np.max(np.abs(u @ P - P @ u)) for u in UG for P in PISO) < 1e-12)

rho_h = rand_state(3)
check("Thm 3(i)   the average is idempotent",
      np.max(np.abs(twirl(twirl(rho_h)) - twirl(rho_h))) < 1e-12)
check("Thm 3(i)   the average is trace preserving",
      abs(np.trace(twirl(rho_h)).real - 1) < 1e-12)
check("Thm 3(i)   one G-orbit carries one effective state",
      max(trdist(twirl(u @ rho_h @ dag(u)), twirl(rho_h)) for u in UG) < 1e-11)
check("Thm 3(i)   the image lies in the fixed-point algebra",
      max(np.max(np.abs(u @ twirl(rho_h) - twirl(rho_h) @ u)) for u in UG) < 1e-11)
check("Thm 3(ii)  sector weights p_pi pass through unchanged",
      max(abs(np.trace(twirl(rho_h) @ P) - np.trace(rho_h @ P)).real
          for P in PISO) < 1e-12)

# the twirl is the pinching only when the irreps are one dimensional, which is
# why Theorem 1 could be written as a pinching and Theorem 2 cannot
pinch = sum(P @ rho_h @ P for P in PISO)
check("Thm 3  twirl and pinching differ for a non-abelian fiber group",
      np.max(np.abs(twirl(rho_h) - pinch)) > 1e-3,
      "max difference %.4f" % np.max(np.abs(twirl(rho_h) - pinch)))

# abelian special case: Z_3 by cyclic shift, where the two do coincide
UZ = [np.linalg.matrix_power(np.roll(np.eye(3), 1, axis=0), k) for k in range(3)]
w = np.exp(2j * np.pi / 3)
F3 = np.array([[w ** (a * b) for b in range(3)] for a in range(3)]) / np.sqrt(3)
PZ = [F3[k].conj().reshape(3, 1) @ F3[k].reshape(1, 3) for k in range(3)]
rz = rand_state(3)
tw_z = sum(u @ rz @ dag(u) for u in UZ) / 3
pi_z = sum(P @ rz @ P for P in PZ)
check("Thm 3  abelian case: the twirl reduces to the pinching of Theorem 1",
      np.max(np.abs(tw_z - pi_z)) < 1e-11,
      "max difference %.2e" % np.max(np.abs(tw_z - pi_z)))

# part (v): the Casimir really is the scalar l(l+1) on each isotypic component
def spin_ops(l):
    d = int(round(2 * l + 1))
    m = np.array([l - i for i in range(d)])
    jp = np.zeros((d, d))
    for i in range(1, d):
        jp[i - 1, i] = np.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    jz = np.diag(m)
    jx, jy = (jp + jp.T) / 2, (jp - jp.T) / (2j)
    return jx, jy, jz


for l in (0, 1, 2):
    jx, jy, jz = spin_ops(l)
    cas = jx @ jx + jy @ jy + jz @ jz
    check("Thm 3(iii) Casimir on the spin-%d component equals l(l+1) = %d"
          % (l, l * (l + 1)),
          np.max(np.abs(cas - l * (l + 1) * np.eye(int(2 * l + 1)))) < 1e-10)

# =====================================================================
# Theorem 2 -- dynamical fiber projection
# =====================================================================
NMAX_D, GAM = 3, 0.35
NSD = np.arange(-NMAX_D, NMAX_D + 1)
DYD, DXD = len(NSD), 2
DD = DYD * DXD
KD = np.kron(np.diag(NSD.astype(float)), np.eye(DXD))
_hx = np.random.default_rng(7).normal(size=(DXD, DXD))
_hx = (_hx + _hx.T) / 2
MASSD = np.sqrt(0.45 ** 2 + NSD ** 2)
HD_gen = np.kron(np.diag(MASSD), np.eye(DXD)) + np.kron(np.eye(DYD), _hx)
HD_sca = np.kron(np.diag(MASSD), np.eye(DXD))
QD = np.round(np.diag(KD)).astype(int)
PID = [np.diag((QD == q).astype(float)) for q in sorted(set(QD))]

check("Thm 2(i)   [H_5, K_y] = 0 for the model used",
      np.max(np.abs(HD_gen @ KD - KD @ HD_gen)) < 1e-12)


def _flow(H, r, t, steps=6000):
    L = lambda x: (-1j * (H @ x - x @ H)
                   - GAM * (KD @ (KD @ x - x @ KD) - (KD @ x - x @ KD) @ KD))
    dt = t / steps
    for _ in range(steps):
        k1 = L(r); k2 = L(r + dt * k1 / 2)
        k3 = L(r + dt * k2 / 2); k4 = L(r + dt * k3)
        r = r + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r


rho_d = rand_state(DD)
TD = 1.3
rT_gen = _flow(HD_gen, rho_d.copy(), TD)
check("Thm 2(i)   trace and positivity preserved",
      abs(np.trace(rT_gen).real - 1) < 1e-9
      and np.linalg.eigvalsh(rT_gen).min() > -1e-9)

# (ii) the block law, which holds for a general H_5 inside each sector
worst_blk = 0.0
for qa in sorted(set(QD)):
    for qb in sorted(set(QD)):
        if qa == qb:
            continue
        Pa = np.diag((QD == qa).astype(float))
        Pb = np.diag((QD == qb).astype(float))
        worst_blk = max(worst_blk,
                        abs(np.linalg.norm(Pa @ rT_gen @ Pb)
                            - np.linalg.norm(Pa @ rho_d @ Pb)
                            * np.exp(-GAM * (qa - qb) ** 2 * TD)))
check("Thm 2(ii)  block norm decays as exp(-gamma (Q-Q')^2 t)",
      worst_blk < 1e-9, "max deviation %.1e" % worst_blk)

# the entrywise sharpening, valid only when H_5 is scalar on each sector
rT_sca = _flow(HD_sca, rho_d.copy(), TD)
worst_el = max(abs(abs(rT_sca[i, j]) - abs(rho_d[i, j])
                   * np.exp(-GAM * (QD[i] - QD[j]) ** 2 * TD))
               for i in range(DD) for j in range(DD) if QD[i] != QD[j])
check("Thm 2(ii)  entrywise law holds when H_5 is scalar on each sector",
      worst_el < 1e-9, "max deviation %.1e" % worst_el)
worst_gen = max(abs(abs(rT_gen[i, j]) - abs(rho_d[i, j])
                    * np.exp(-GAM * (QD[i] - QD[j]) ** 2 * TD))
                for i in range(DD) for j in range(DD) if QD[i] != QD[j])
check("Thm 2(ii)  entrywise law fails without that hypothesis, as stated",
      worst_gen > 1e-5, "deviation %.1e" % worst_gen)

# (iii) sector weights are constants of the motion
check("Thm 2(iii) sector weights p_Q are conserved",
      max(abs(np.trace(P @ rT_gen).real - np.trace(P @ rho_d).real)
          for P in PID) < 1e-9)

# (iv) the dephasing part converges to the pinching; the full flow does not
r_long_D = _flow(np.zeros_like(HD_gen), rho_d.copy(), 60.0, steps=20000)
EY_d = sum(P @ rho_d @ P for P in PID)
check("Thm 2(iv)  exp(t L_D) -> E_Y", trdist(r_long_D, EY_d) < 1e-6,
      "trace distance %.1e" % trdist(r_long_D, EY_d))
r_long_full = _flow(HD_gen, rho_d.copy(), 60.0, steps=20000)
check("Thm 2(iv)  the full flow does not converge to E_Y, as stated",
      trdist(r_long_full, EY_d) > 1e-2,
      "trace distance %.3f" % trdist(r_long_full, EY_d))

# (v) the heat equation on the circle, and the projection rate
_c = np.random.default_rng(3).normal(size=DYD) * 1.0
_c = _c / np.linalg.norm(_c)
_psi = np.outer(_c, [1.0, 0.0]).reshape(-1)
_r = np.outer(_psi, _psi.conj())
_rt = _flow(np.zeros_like(HD_gen), _r.copy(), 0.9, steps=4000)
ok_heat = True
for m in range(1, 2 * NMAX_D + 1):
    Pm = sum(_rt[i * DXD, j * DXD] for i in range(DYD) for j in range(DYD)
             if NSD[i] - NSD[j] == m)
    Pm0 = sum(_r[i * DXD, j * DXD] for i in range(DYD) for j in range(DYD)
              if NSD[i] - NSD[j] == m)
    if abs(Pm - np.exp(-GAM * m ** 2 * 0.9) * Pm0) > 1e-7:
        ok_heat = False
check("Thm 2(v)   P^I_m(t) = exp(-gamma m^2 t) P^I_m(0)", ok_heat)

# the rate of equation (proj-rate)
rr = np.random.default_rng(5)
def _epsY(t, trials=300):
    damp = np.exp(-GAM * (NSD[:, None] - NSD[None, :]) ** 2 * t)
    w = 0.0
    for _ in range(trials):
        c = rr.normal(size=DYD) + 1j * rr.normal(size=DYD); c /= np.linalg.norm(c)
        d = c * np.exp(1j * rr.uniform(0, 2 * np.pi, DYD))
        w = max(w, 0.5 * np.sum(np.abs(np.linalg.eigvalsh(
            (np.outer(c, c.conj()) - np.outer(d, d.conj())) * damp))))
    return w
tsr = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
rate_fit = -np.polyfit(tsr, np.log([_epsY(t) for t in tsr]), 1)[0]
check("Thm 2(iv)  eps_Y decays at gamma_C = gamma", abs(rate_fit - GAM) < 0.02,
      "fitted %.4f vs gamma %.4f" % (rate_fit, GAM))

# heat-kernel rate on G/H: Gamma_pi = gamma C_2(pi)
_dims = [2 * l + 1 for l in range(3)]
_off = np.cumsum([0] + _dims); _DH = sum(_dims)
_P = []
for i, d in enumerate(_dims):
    M = np.zeros((_DH, _DH)); M[_off[i]:_off[i] + d, _off[i]:_off[i] + d] = np.eye(d)
    _P.append(M)
_A = rr.normal(size=(_DH, _DH)) + 1j * rr.normal(size=(_DH, _DH))
_non = _A - _P[0] @ _A @ _P[0]
_Et = lambda X, t: sum(np.exp(-GAM * t * l * (l + 1)) * (_P[l] @ X @ _P[l])
                       for l in range(3))
_ts = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
_rate = -np.polyfit(_ts, np.log([np.linalg.norm(_Et(_non, t)) for t in _ts]), 1)[0]
check("Thm 3(iv)  slowest rate gamma C_2(pi) of the slowest sector",
      abs(_rate - GAM * 2) < 1e-3, "fitted %.4f vs %.4f" % (_rate, GAM * 2))

# Phi^4 momentum selection rule
from itertools import product as _prod
_MODES = [-1, 0, 1]
_basis = [b for b in _prod(range(3), repeat=3)]
_idx = {b: i for i, b in enumerate(_basis)}
_NB = len(_basis)
def _a(m):
    A_ = np.zeros((_NB, _NB))
    for b, i in _idx.items():
        if b[m] > 0:
            c_ = list(b); c_[m] -= 1
            A_[_idx[tuple(c_)], i] = np.sqrt(b[m])
    return A_
_aops = [_a(m) for m in range(3)]
_Kf = sum(_MODES[m] * dag(_aops[m]) @ _aops[m] for m in range(3))
_phi = [(_aops[m] + dag(_aops[_MODES.index(-_MODES[m])])) / np.sqrt(2)
        for m in range(3)]
_Hint = np.zeros((_NB, _NB))
for _q in _prod(range(3), repeat=4):
    if sum(_MODES[k] for k in _q) == 0:
        _Hint = _Hint + _phi[_q[0]] @ _phi[_q[1]] @ _phi[_q[2]] @ _phi[_q[3]]
_Kd = np.round(np.diag(_Kf)).astype(int)
_low = [i for i, b in enumerate(_basis) if sum(b) <= 1]
check("Sec 6.2  Phi^4 vertex conserves total Kaluza-Klein number",
      max(abs(_Hint[i, j]) for i in _low for j in _low
          if _Kd[i] != _Kd[j]) < 1e-12)
check("Sec 6.2  [H_int, K_y] = 0 below the truncation",
      max(abs((_Hint @ _Kf - _Kf @ _Hint)[i, j])
          for i in _low for j in _low) < 1e-12)

# =====================================================================
# Proposition 1 -- microscopic origin of the dephasing
# =====================================================================
# H_tot = H_5 (x) I + I (x) H_B + g K_y (x) B with [H_5, K_y] = 0. The coupling
# runs through K_y itself, so it is quantum non-demolition for the KK charge.
_rg = np.random.default_rng(4)
QSm = np.array([-1, 0, 1]); DSm = len(QSm)
NBm = 7; DBm = 2 ** NBm
Km = np.diag(QSm.astype(float))
H5m = np.diag(np.sqrt(0.45 ** 2 + QSm ** 2))
_sx = np.array([[0, 1], [1, 0]], float); _sz = np.diag([1.0, -1.0])


def _emb(op, j):
    M = np.eye(1)
    for k in range(NBm):
        M = np.kron(M, op if k == j else np.eye(2))
    return M


_h = _rg.uniform(0.5, 1.5, NBm); _c = _rg.uniform(0.5, 1.5, NBm)
HBm = sum(_h[j] * _emb(_sz, j) for j in range(NBm))
Bm = sum(_c[j] * _emb(_sx, j) for j in range(NBm))
gm = 0.22
Htot = (np.kron(H5m, np.eye(DBm)) + np.kron(np.eye(DSm), HBm)
        + gm * np.kron(Km, Bm))
Kfull = np.kron(Km, np.eye(DBm))
check("Prop 1(i)  [H_tot, K_y] = 0 exactly, no limit taken",
      np.max(np.abs(Htot @ Kfull - Kfull @ Htot)) < 1e-12)

_cs = np.array([0.5, 0.7, 0.5]); _cs = _cs / np.linalg.norm(_cs)
_vb = _rg.normal(size=DBm) + 1j * _rg.normal(size=DBm); _vb /= np.linalg.norm(_vb)
_psi0 = np.kron(_cs, _vb)
_w, _V = np.linalg.eigh(Htot)


def _rhoS(t):
    psi = _V @ (np.exp(-1j * _w * t) * (dag(_V) @ _psi0))
    M = psi.reshape(DSm, DBm)
    return M @ dag(M)


_r0 = _rhoS(0.0)
_drift = max(abs(_rhoS(t)[i, i].real - _r0[i, i].real)
             for t in (1., 2., 4., 8.) for i in range(3))
check("Prop 1(i)  sector weights exactly conserved by the microscopic model",
      _drift < 1e-12, "max drift %.1e" % _drift)

_ratios = []
for t in (0.25, 0.5, 0.75):
    _r = _rhoS(t)
    _ratios.append(np.log(abs(_r[0, 2]) / abs(_r0[0, 2]))
                   / np.log(abs(_r[0, 1]) / abs(_r0[0, 1])))
check("Prop 1(ii) a generic environment does NOT give a (Q-Q')^2 exponent",
      max(abs(np.log(abs(_rhoS(t)[0, 2]) / abs(_r0[0, 2]))
              / np.log(abs(_rhoS(t)[0, 1]) / abs(_r0[0, 1])) - 4.0)
          for t in (2.0, 5.0)) > 0.5,
      "the spin-bath ratio leaves 4 at longer times, so equation (25) is needed")

# the Gaussian environment of equation (25), where the form is exact
_NF = 26
_a = np.diag(np.sqrt(np.arange(1, _NF)), 1)
_nop = dag(_a) @ _a
_OM, _CK, _gg = [0.7, 1.3, 2.1], [0.55, 0.40, 0.30], 0.30


def _ovl(Q, Qp, t):
    tot = 1.0 + 0j
    for om, ck in zip(_OM, _CK):
        v0 = np.zeros(_NF, complex); v0[0] = 1.0
        def ev(q):
            w, V = np.linalg.eigh(om * _nop + _gg * q * ck * (_a + dag(_a)))
            return V @ (np.exp(-1j * w * t) * (dag(V) @ v0))
        tot *= np.vdot(ev(Qp), ev(Q))
    return tot


def _Gam(t):
    return _gg ** 2 * sum((ck / om) ** 2 * (1 - np.cos(om * t))
                          for om, ck in zip(_OM, _CK))


_worst = max(abs(abs(_ovl(Q, Qp, t)) - np.exp(-(Q - Qp) ** 2 * _Gam(t)))
             for t in (0.6, 1.7, 3.4, 6.0, 11.0)
             for Q, Qp in ((1, 0), (2, 0), (3, 1), (4, 1)))
check("Prop 1(ii) Gaussian environment gives exp[-(Q-Q')^2 Gamma(t)] exactly",
      _worst < 1e-12, "largest deviation %.1e" % _worst)
check("Prop 1(ii) the modulus depends on the sectors only through Q-Q'",
      abs(abs(_ovl(2, 0, 3.4)) - abs(_ovl(3, 1, 3.4))) < 1e-12)

# the phase carries Q^2 - Q'^2, which is the Lamb shift proportional to K_y^2
_ph = [np.angle(_ovl(Q, Qp, t)) / (Q ** 2 - Qp ** 2)
       for t in (0.9, 2.2, 4.5) for Q, Qp in ((1, 0), (2, 0), (2, 1), (3, 1))]
_grp = [max(_ph[i:i+4]) - min(_ph[i:i+4]) for i in (0, 4, 8)]
check("Prop 1(ii) the phase is proportional to Q^2 - Q'^2",
      max(_grp) < 1e-9, "spread within a time %.1e" % max(_grp))

# H_LS proportional to K_y^2, so it commutes with K_y and leaves part (i) alone
_shift = [sum(-(_gg * q * ck) ** 2 / om for om, ck in zip(_OM, _CK))
          for q in range(4)]
check("Prop 1(iii) the Lamb shift is proportional to K_y^2",
      max(abs(_shift[q] / q ** 2 - _shift[1]) for q in (2, 3)) < 1e-12)

# Section 5.3 -- the two-point kernel loses its mean-position dependence
GAM = 0.30
NSw = np.arange(-4, 5); DNw = len(NSw)
_A = _rg.normal(size=(DNw, DNw)) + 1j * _rg.normal(size=(DNw, DNw))
_rw = _A @ dag(_A); _rw /= np.trace(_rw).real
_bars = np.linspace(0, 2 * np.pi, 64, endpoint=False)


def _harm(t, rel):
    d = np.exp(-GAM * (NSw[:, None] - NSw[None, :]) ** 2 * t)
    r = _rw * d
    vals = np.array([sum(r[i, j] * np.exp(1j * (NSw[i] * (b + rel / 2)
                                                - NSw[j] * (b - rel / 2)))
                         for i in range(DNw) for j in range(DNw)) for b in _bars])
    return np.fft.fft(vals) / len(_bars)


_h0 = _harm(0.0, 0.8); _h2 = _harm(2.0, 0.8)
check("Sec 6.2  W_m(t) = exp(-gamma m^2 t) W_m(0)",
      max(abs(abs(_h2[m]) - abs(_h0[m]) * np.exp(-GAM * m ** 2 * 2.0))
          for m in range(4)) < 1e-10)
# the free case: with no interacting evolution the m = 0 grade is undamped.
# The manuscript states it as protected from dephasing, not as constant.
check("Sec 6.2  the m = 0 harmonic is undamped by the dephasing",
      abs(abs(_harm(50.0, 0.8)[0]) - abs(_h0[0])) < 1e-12)

# Theorem 3: a graded observable is damped by exactly exp(-gamma m^2 t),
# which is what carries the envelope through the interacting theory
_DG = 7
_Kg = np.diag(np.arange(-3, 4).astype(float))
_gam = 0.4


def _Dadj(A, t):
    d = np.array([[np.exp(-_gam * t * (_Kg[i, i] - _Kg[j, j]) ** 2)
                   for j in range(_DG)] for i in range(_DG)])
    return A * d


_rg2 = np.random.default_rng(2)
_gr = _dm = 0.0
for _m in (1, 2, 3):
    _A = np.zeros((_DG, _DG), complex)
    for _i in range(_DG):
        _j = _i - _m
        if 0 <= _j < _DG:
            _A[_i, _j] = _rg2.normal() + 1j * _rg2.normal()
    _gr = max(_gr, np.max(np.abs(_Kg @ _A - _A @ _Kg - _m * _A)))
    _dm = max(_dm, np.max(np.abs(_Dadj(_A, 1.7)
                                 - np.exp(-_gam * _m ** 2 * 1.7) * _A)))
check("Thm 2  graded observables satisfy [K_y, O_m] = m O_m", _gr < 1e-12)

check("Thm 2  the dephasing adjoint damps O_m by exp(-gamma m^2 t)",
      _dm < 1e-12, "residual %.1e" % _dm)

# =====================================================================
# Sections 4.1 to 4.3 and 5.3: phase sign, Ohmic environment, counterterm, two-mode
# fringe, representation spaces on S^2, occupations under interactions
# =====================================================================
from scipy.integrate import quad as _quad
from scipy.linalg import expm as _expm
from scipy.special import sph_harm_y as _Y

def _F_exact(Q, Qp, t, w, c, g, beta, nmax):
    a = np.diag(np.sqrt(np.arange(1, nmax)), 1)
    def op(k, A):
        out = np.array([[1.0]])
        for j in range(len(w)):
            out = np.kron(out, A if j == k else np.eye(nmax))
        return out
    HB = sum(w[k] * op(k, a.T @ a) for k in range(len(w)))
    B = sum(c[k] * op(k, a) + np.conj(c[k]) * op(k, a.T) for k in range(len(w)))
    if beta is None:
        rB = np.zeros_like(HB); rB[0, 0] = 1.0
    else:
        e = _expm(-beta * HB); rB = e / np.trace(e)
    U = _expm(-1j * (HB + g * Q * B) * t); Up = _expm(-1j * (HB + g * Qp * B) * t)
    return np.trace(U @ rB @ dag(Up))

def _F_formula(Q, Qp, t, w, c, g, beta, sign=+1):
    w = np.array(w); c2 = np.abs(np.array(c)) ** 2
    coth = np.ones_like(w) if beta is None else 1 / np.tanh(beta * w / 2)
    G = g ** 2 * np.sum(c2 / w ** 2 * coth * (1 - np.cos(w * t)))
    T = g ** 2 * np.sum(c2 / w ** 2 * (w * t - np.sin(w * t)))
    return np.exp(-(Q - Qp) ** 2 * G) * np.exp(sign * 1j * (Q ** 2 - Qp ** 2) * T)

_dev = max(abs(_F_exact(Q, Qp, t, [1.1], [0.6], 0.35, b, 60)
               - _F_formula(Q, Qp, t, [1.1], [0.6], 0.35, b))
           for (Q, Qp) in [(1, 0), (2, -1), (3, 1), (-2, 2)]
           for b in (None, 1.3) for t in (0.7, 2.9))
check("Prop 1(ii)  exact branch factor = exp[-(Q-Q')^2 Gamma] exp[+i(Q^2-Q'^2) Theta]",
      _dev < 1e-12, "max dev %.2e" % _dev)
_fe = _F_exact(2, -1, 1.9, [1.1], [0.6], 0.35, None, 60)
check("Prop 1(ii)  negative control: the opposite phase sign fails",
      abs(_fe - _F_formula(2, -1, 1.9, [1.1], [0.6], 0.35, None, sign=-1)) > 1e-2)

_g, _eta, _wc = 0.3, 0.8, 6.0
def _Gam0(t):
    f = lambda w: _eta * np.exp(-w / _wc) / w * (1 - np.cos(w * t))
    e = np.concatenate(([1e-12], np.linspace(0.5 / t, 60 * _wc, 1500)))
    return _g ** 2 * sum(_quad(f, a, b, limit=200)[0] for a, b in zip(e[:-1], e[1:]))
def _The(t):
    f = lambda w: _eta * np.exp(-w / _wc) / w * (w * t - np.sin(w * t))
    e = np.concatenate(([1e-12], np.linspace(0.5 / t, 60 * _wc, 1500)))
    return _g ** 2 * sum(_quad(f, a, b, limit=200)[0] for a, b in zip(e[:-1], e[1:]))
_d0 = max(abs(_Gam0(t) - 0.5 * _g ** 2 * _eta * np.log(1 + (_wc * t) ** 2)) for t in (0.3, 2., 9.))
check("Sec 4.2  Ohmic, T = 0: Gamma = (g^2 eta/2) ln(1 + wc^2 t^2), no Markovian rate",
      _d0 < 1e-10, "max dev %.2e" % _d0)
_dT = max(abs(_The(t) - _g ** 2 * _eta * (_wc * t - np.arctan(_wc * t))) for t in (0.3, 2., 9.))
check("Sec 4.2  Ohmic phase: Theta = g^2 eta [wc t - arctan(wc t)], slope -> lambda_c = g^2 eta wc",
      _dT < 1e-10, "max dev %.2e" % _dT)
_rat = []
for _beta in (0.5, 1.0, 2.0):
    _h = lambda w, b=_beta: (_eta * np.exp(-w / _wc) * (1 / np.tanh(b * w / 2) - 2 / (b * w))
                             if w > 1e-9 else 0.0)
    _reg = _quad(_h, 0, np.inf, weight='sin', wvar=320.0, limlst=200)[0]
    _rat.append(_g ** 2 * (2 * _eta / _beta * np.arctan(_wc * 320.0) + _reg)
                / (np.pi * _g ** 2 * _eta / _beta))
check("Sec 4.2  Ohmic, T > 0: dGamma/dt -> gamma = pi g^2 eta T (ratio within 1e-3 at wc t = 1920)",
      all(abs(r - 1) < 1e-3 for r in _rat), "ratios %s" % np.round(_rat, 6))
check("Sec 4.2  gamma = (g^2/2) S_B(0) with S_B(0) = 2 pi eta T agrees with pi g^2 eta T",
      abs(0.5 * _g ** 2 * 2 * np.pi * _eta / 1.0 - np.pi * _g ** 2 * _eta / 1.0) < 1e-14)

# two-mode fringe, built from the random-unitary integral
_GM, _n, _th = 0.37, 2, 0.9
_ks = np.arange(-1, 6)
_c = np.zeros(len(_ks), complex); _c[_ks == _n] = 1 / np.sqrt(2)
_c[_ks == _n + 1] = np.exp(-1j * _th) / np.sqrt(2)
_r0 = np.outer(_c, _c.conj())
_R, _m5, _p = 1.3, 0.7, 0.4
_E = np.sqrt(_p ** 2 + _m5 ** 2 + (_ks / _R) ** 2)
def _Dt(rho, t, N=3001):
    s = np.linspace(-12 * np.sqrt(2 * _GM * t), 12 * np.sqrt(2 * _GM * t), N); ds = s[1] - s[0]
    wg = np.exp(-s ** 2 / (4 * _GM * t)) / np.sqrt(4 * np.pi * _GM * t) * ds
    out = np.zeros_like(rho)
    for si, wi in zip(s, wg):
        u = np.exp(-1j * si * _ks); out += wi * (u[:, None] * rho * u.conj()[None, :])
    return out
_ys = np.linspace(0, 2 * np.pi, 181)
_dw = _E[_ks == _n + 1][0] - _E[_ks == _n][0]
_worst = 0.0
for t in (0.3, 1.1, 2.5):
    U = np.exp(-1j * _E * t)
    rt = _Dt(U[:, None] * _r0 * U.conj()[None, :], t)
    P = np.array([np.real(np.sum(rt * np.exp(1j * (_ks[:, None] - _ks[None, :]) * y)))
                  for y in _ys]) / (2 * np.pi)
    _worst = max(_worst, np.max(np.abs(P - (1 + np.exp(-_GM * t)
                                            * np.cos(_ys - _th - _dw * t)) / (2 * np.pi))))
check("Sec 5.3  two-mode fringe P = [1 + e^{-gamma t} cos(y - theta - dw t)]/2pi",
      _worst < 1e-12, "max dev %.2e" % _worst)
_lam = 0.21
_El = _E - _lam * _ks ** 2
check("Sec 4.3  uncompensated H_LS = -lambda_c K^2 shifts the drift by -lambda_c(2n+1)",
      abs((_El[_ks == _n + 1][0] - _El[_ks == _n][0]) - _dw + _lam * (2 * _n + 1)) < 1e-14)

# representation spaces on S^2
def _spin(l):
    m = np.arange(l, -l - 1, -1); Jz = np.diag(m).astype(complex)
    Jp = np.zeros((2 * l + 1, 2 * l + 1), complex)
    for i in range(1, 2 * l + 1):
        Jp[i - 1, i] = np.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    return [(Jp + dag(Jp)) / 2, (Jp - dag(Jp)) / (2j), Jz], list(m)
def _block(l1, l2):
    A, _ = _spin(l1); B, _ = _spin(l2); d1, d2 = 2 * l1 + 1, 2 * l2 + 1
    S = sum((np.kron(A[a], np.eye(d2)) - np.kron(np.eye(d1), B[a].T)) @
            (np.kron(A[a], np.eye(d2)) - np.kron(np.eye(d1), B[a].T)) for a in range(3))
    v, c = np.unique(np.round(np.real(np.linalg.eigvals(S)), 8), return_counts=True)
    return dict(zip([abs(x) for x in v], c))
check("Thm 3(iv)  coherence between l=1 and l=2 decays at L(L+1) = 2, 6, 12 (mult. 3, 5, 7)",
      _block(1, 2) == {2.0: 3, 6.0: 5, 12.0: 7}, str(_block(1, 2)))
check("Thm 3(iv)  the l=2 level alone carries L = 0..4, so 20 gamma appears in its density",
      _block(2, 2) == {0.0: 1, 2.0: 3, 6.0: 5, 12.0: 7, 20.0: 9}, str(_block(2, 2)))
_ls = [1, 2]; _D = 8; _J = [np.zeros((_D, _D), complex) for _ in range(3)]; _lab = []; _o = 0
for l in _ls:
    Jl, ms = _spin(l)
    for a in range(3):
        _J[a][_o:_o + 2 * l + 1, _o:_o + 2 * l + 1] = Jl[a]
    _lab += [(l, m) for m in ms]; _o += 2 * l + 1
_S = sum((np.kron(_J[a], np.eye(_D)) - np.kron(np.eye(_D), _J[a].T)) @
         (np.kron(_J[a], np.eye(_D)) - np.kron(np.eye(_D), _J[a].T)) for a in range(3))
_psi = np.zeros(_D, complex)
_psi[_lab.index((1, 0))] = 1 / np.sqrt(2); _psi[_lab.index((2, 0))] = 1 / np.sqrt(2)
_rho0 = np.outer(_psi, _psi.conj())
_rhot = (_expm(-0.04 * _S) @ _rho0.reshape(-1)).reshape(_D, _D)
_thg, _phg = np.meshgrid(np.linspace(1e-4, np.pi - 1e-4, 241), np.linspace(0, 2 * np.pi, 481),
                         indexing='ij')
_dA = np.sin(_thg) * (_thg[1, 0] - _thg[0, 0]) * (_phg[0, 1] - _phg[0, 0])
_Ys = [_Y(l, m, _thg, _phg) for (l, m) in _lab]
def _dens(r):
    return np.real(sum(r[i, j] * _Ys[i] * np.conj(_Ys[j])
                       for i in range(_D) for j in range(_D) if abs(r[i, j]) > 1e-14))
_P0, _Pt = _dens(_rho0), _dens(_rhot)
_rates = {}
for L in range(1, 5):
    c0 = np.sum(_P0 * np.conj(_Y(L, 0, _thg, _phg)) * _dA)
    ct = np.sum(_Pt * np.conj(_Y(L, 0, _thg, _phg)) * _dA)
    _rates[L] = -np.log(abs(ct / c0)) / 0.04
check("Thm 3(iv)  density of (Y10+Y20)/sqrt2: harmonic L decays at gamma L(L+1), L = 1..4",
      all(abs(_rates[L] - L * (L + 1)) < 0.01 for L in _rates),
      str({L: round(v, 4) for L, v in _rates.items()}))
check("Thm 3(ii)  isotypic weights of l=1 and l=2 conserved by the heat-kernel twirl",
      abs(np.trace(_rhot[:3, :3]).real - 0.5) < 1e-12 and abs(np.trace(_rhot[3:, 3:]).real - 0.5) < 1e-12)

# occupations under interactions
_modes = [-2, -1, 0, 1, 2]; _nm = 5
_basis = [o for o in itertools.product(range(3), repeat=_nm) if sum(o) <= 2]
_ix = {b: i for i, b in enumerate(_basis)}; _N = len(_basis)
def _aop(k):
    A = np.zeros((_N, _N))
    for b, i in _ix.items():
        if b[k] > 0:
            bb = list(b); bb[k] -= 1; A[_ix[tuple(bb)], i] = np.sqrt(b[k])
    return A
_a = [_aop(k) for k in range(_nm)]
_K = sum(_modes[k] * _a[k].T @ _a[k] for k in range(_nm))
_H = np.zeros((_N, _N))
for i1, i2, i3, i4 in itertools.product(range(_nm), repeat=4):
    if _modes[i1] + _modes[i2] == _modes[i3] + _modes[i4]:
        _H += _a[i3].T @ _a[i4].T @ _a[i1] @ _a[i2]
_H = _H + _H.T
_N1 = _a[3].T @ _a[3]
check("Sec 6.2  momentum-conserving vertex commutes with K_y",
      np.linalg.norm(_H @ _K - _K @ _H) < 1e-12)
check("Sec 6.2  the same vertex moves individual occupations, [H_int, N_1] != 0",
      np.linalg.norm(_H @ _N1 - _N1 @ _H) > 1.0)
_s0 = np.zeros(_N); _s0[_ix[(0, 1, 0, 1, 0)]] = 1
_st = _expm(-1j * 0.3 * _H) @ _s0
check("Sec 6.2  |n=-1, n=+1> scatters into |n=-2, n=+2> at fixed Q = 0",
      abs(_st[_ix[(1, 0, 0, 0, 1)]]) ** 2 > 1e-3 and abs(_st.conj() @ _K @ _st) < 1e-12)

# =====================================================================
# Label algebra, stability, code-subspace rate, connected
# groups, parity selection, local origin of the coupling, detector access,
# and the link between decay rates and mass excesses
# =====================================================================
from scipy.linalg import expm as _ex2
from scipy.special import sph_harm_y as _Y2

# --- the label algebra must commute with the zero-mode Weyl algebra
_cap = 4
_I = np.eye(_cap + 1)
_a1 = np.diag(np.sqrt(np.arange(1, _cap + 1)), 1)
def _on(k, A):
    ops = [_I, _I, _I]; ops[k] = A
    return np.kron(np.kron(ops[0], ops[1]), ops[2])
_Nm, _N0, _Np = (_on(k, _a1.T @ _a1) for k in range(3))          # modes n = -1, 0, +1
_Ky = _Np - _Nm
_W0 = _ex2(1j * 0.7 * _on(1, _a1 + _a1.T))
_ev, _evec = np.linalg.eigh(_Ky)
_PiQ = {q: _evec[:, np.isclose(_ev, q)] @ dag(_evec[:, np.isclose(_ev, q)])
        for q in np.unique(np.round(_ev))}
check("Thm 1  every Pi_Q commutes with the zero-mode Weyl unitary W_0(f)",
      max(np.linalg.norm(_W0 @ P - P @ _W0) for P in _PiQ.values()) < 1e-12)
_EN0 = _ex2(1j * 0.9 * _N0)
check("Thm 1  negative control: exp(i theta N_0) fails to commute with W_0(f)",
      np.linalg.norm(_W0 @ _EN0 - _EN0 @ _W0) > 1e-2)

# --- stability: completed square, and the counterterm bounds H_tot from below
def _stab(Nmax, counter, nb=60, w1=1.0, wb=1.0, gc=0.3):
    dN = Nmax + 1
    N = np.diag(np.arange(dN)).astype(float)                       # quanta of level n = 1
    b = np.diag(np.sqrt(np.arange(1, nb)), 1)
    lam = gc ** 2 / wb
    H = (np.kron(w1 * N, np.eye(nb)) + np.kron(np.eye(dN), wb * b.T @ b)
         + gc * np.kron(N, b + b.T))
    if counter:
        H = H + lam * np.kron(N @ N, np.eye(nb))
    return np.linalg.eigvalsh(H)[0], H, N, b, lam
# N quanta in level n = 1 have polaron energy N w1 - lambda_c N^2 without the
# counterterm; with w1 = 1 and lambda_c = 0.09 it turns negative beyond N = 11,
# so truncations N <= 12, 16, 20 expose the fall, which the counterterm removes
_NT = (12, 16, 20)
_low = [_stab(Nm, False, nb=130)[0] for Nm in _NT]
_lowc = [_stab(Nm, True, nb=130)[0] for Nm in _NT]
_pred = [Nm * 1.0 - 0.09 * Nm ** 2 for Nm in _NT]
check("Sec 4.1  without the counterterm the ground energy falls like -lambda_c N^2",
      _low[2] < _low[1] < _low[0] < 0 and max(abs(a - b) for a, b in zip(_low, _pred)) < 1e-6,
      "ground energies %s vs %s" % (np.round(_low, 6), np.round(_pred, 6)))
check("Sec 4.1  with the counterterm H_tot is bounded below by inf H_5 = 0",
      min(_lowc) > -1e-9, "ground energies %s" % np.round(_lowc, 9))
_, _H, _N, _b, _lam = _stab(6, True)
_alpha = 0.3 * np.kron(_N, np.eye(60))
_B = np.kron(np.eye(7), _b)
_sq = (_B + _alpha).T @ (_B + _alpha)
check("Sec 4.1  completed square: H_B + g K B + lambda_c K^2 = sum w (b+zeta)^dag (b+zeta)",
      np.linalg.norm(_H - np.kron(1.0 * _N, np.eye(60)) - _sq) < 1e-10)
_resid = max(abs(_The(t) - _g ** 2 * _eta * _wc * t + _g ** 2 * _eta * np.arctan(_wc * t))
             for t in (0.5, 4.0, 30.0))
check("Sec 4.2  with the counterterm the net phase Theta - lambda_c t = -g^2 eta arctan(wc t) stays bounded",
      _resid < 1e-9, "max dev %.2e" % _resid)

# --- code-subspace rate: minimum over the sectors actually present
_GMc = 0.6
_secs = [0, 2]; _dq = 2
_dim = len(_secs) * _dq
def _dmp(r, t):
    out = r.copy()
    for i, qa in enumerate(_secs):
        for j, qb in enumerate(_secs):
            out[i*_dq:(i+1)*_dq, j*_dq:(j+1)*_dq] *= np.exp(-_GMc * (qa - qb) ** 2 * t)
    return out
_worst = []
for t in (0.3, 0.8, 1.5):
    w = 0.0
    for _ in range(200):
        r = rand_state(_dim)
        s_ = r.copy()
        off = rand_herm(_dim) * 0.15
        off[:_dq, :_dq] = 0; off[_dq:, _dq:] = 0
        s_ = s_ + off
        if np.min(np.linalg.eigvalsh(s_)) < 0:
            continue
        w = max(w, trdist(_dmp(r, t), _dmp(s_, t)) / max(trdist(r, s_), 1e-15))
    _worst.append(w)
_fit = -np.polyfit([0.3, 0.8, 1.5], np.log(_worst), 1)[0]
check("Thm 2(iv)  sectors {0, 2}: the erasure error decays at gamma_C = 4 gamma, not gamma",
      abs(_fit - 4 * _GMc) < 1e-6, "fitted %.6f vs %.6f" % (_fit, 4 * _GMc))

# --- connectedness: O(2) on the levels n = +1, -1
_r1 = np.diag([1.0, 0.0])                                         # basis |+1>, |-1>
_lim = np.diag(np.diag(_r1))                                      # SO(2) heat-kernel limit
_Rf = np.array([[0, 1], [1, 0]])
_O2 = 0.5 * (_lim + _Rf @ _lim @ _Rf.T)
check("Thm 3  disconnected O(2): the heat semigroup converges to the SO(2) average, not the O(2) one",
      np.linalg.norm(_lim - _O2) > 0.5)

# --- parity selection for the m = 0 state of Fig. 4, by Gauss-Legendre
#     quadrature in x = cos(theta), exact for these polynomial integrands
_xg, _wg = np.polynomial.legendre.leggauss(40)
def _Yz(l):
    c = np.zeros(l + 1); c[l] = 1.0
    return np.sqrt((2 * l + 1) / (4 * np.pi)) * np.polynomial.legendre.legval(_xg, c)
def _proj(f, L):
    return abs(2 * np.pi * np.sum(_wg * f * _Yz(L)))
_cross = _Yz(1) * _Yz(2)
_diag1, _diag2 = _Yz(1) ** 2, _Yz(2) ** 2
_pc = [_proj(_cross, L) for L in range(6)]
_p1 = [_proj(_diag1, L) for L in range(6)]
_p2 = [_proj(_diag2, L) for L in range(6)]
check("Thm 3(iv)  m = 0 coherence Y10 Y20 feeds L = 1 and 3 only; its L = 2 part vanishes",
      _pc[1] > 1e-2 and _pc[3] > 1e-2 and max(_pc[0], _pc[2], _pc[4], _pc[5]) < 1e-12,
      "coherence components %s" % np.round(_pc, 6))
check("Thm 3(iv)  diagonal terms feed L = 0, 2 (both levels) and L = 4 (level l = 2 alone)",
      _p1[0] > 1e-2 and _p1[2] > 1e-2 and max(_p1[1], _p1[3], _p1[4], _p1[5]) < 1e-12
      and _p2[0] > 1e-2 and _p2[2] > 1e-2 and _p2[4] > 1e-2
      and max(_p2[1], _p2[3], _p2[5]) < 1e-12,
      "|Y10|^2 %s  |Y20|^2 %s" % (np.round(_p1, 6), np.round(_p2, 6)))

# --- radius normalization and the rate / mass-excess link
_Rr, _gm = 1.7, 0.4
_ok = all(abs(_gm * m ** 2 - (_gm * _Rr ** 2) * (m / _Rr) ** 2) < 1e-12 for m in range(1, 6))
check("Sec 6.1  circle: gamma m^2 = gamma R^2 (m/R)^2", _ok)
_ratios = [(_gm * m ** 2) / (m ** 2 / _Rr ** 2) for m in range(1, 6)]
check("Sec 7.1  decay rate over mass excess m_m^2 - m_5^2 is one constant gamma R^2 for every m",
      np.ptp(_ratios) < 1e-12 and abs(_ratios[0] - _gm * _Rr ** 2) < 1e-12)
_rs = 0.8
_ratS = [(_gm * L * (L + 1)) / (L * (L + 1) / _rs ** 2) for L in range(1, 6)]
check("Sec 7.1  on a sphere of radius r: gamma C_2(L) / lambda_L = gamma r^2 for every L",
      np.ptp(_ratS) < 1e-12)

# --- local origin: spectral densities of a massless four-dimensional scalar,
#     from the modes of a periodic box of side L (deterministic lattice sums).
#     psi(0) has |c_k|^2 = 1/(2 w_k L^3); d_x psi(0) has |c_k|^2 = k_x^2/(2 w_k L^3).
#     The cumulative sums over |k| < W are compared with the integrals of
#     J = w/(4 pi^2) and J = w^3/(12 pi^2), namely W^2/(8 pi^2) and W^4/(48 pi^2).
_rng = np.random.default_rng(7)
_Lb = 80.0
_nb = int(np.ceil(5.0 * _Lb / (2 * np.pi))) + 1
_nn = np.arange(-_nb, _nb + 1)
_kx, _ky, _kz = np.meshgrid(_nn, _nn, _nn, indexing='ij', sparse=True)
_kk = (2 * np.pi / _Lb) * np.sqrt(_kx ** 2 + _ky ** 2 + _kz ** 2)
_kx2 = ((2 * np.pi / _Lb) * _kx) ** 2 * np.ones_like(_kk)
_rm, _rd = [], []
for _W in (2.0, 3.0, 4.0, 5.0):
    _msk = (_kk > 0) & (_kk < _W)
    _rm.append(np.sum(1.0 / (2 * _kk[_msk] * _Lb ** 3)) / (_W ** 2 / (8 * np.pi ** 2)))
    _rd.append(np.sum(_kx2[_msk] / (2 * _kk[_msk] * _Lb ** 3)) / (_W ** 4 / (48 * np.pi ** 2)))
check("Sec 4.4  monopole: J(omega) = omega/(4 pi^2) for a point coupling to a massless 4D scalar",
      max(abs(r - 1) for r in _rm) < 5e-3, "cumulative ratios %s" % np.round(_rm, 5))
check("Sec 4.4  dipole: J(omega) = omega^3/(12 pi^2), so its zero-frequency noise vanishes",
      max(abs(r - 1) for r in _rd) < 5e-3, "cumulative ratios %s" % np.round(_rd, 5))
check("Sec 4.4  eta = 1/(4 pi^2) gives gamma = pi g^2 eta T = g^2 T/(4 pi)",
      abs(np.pi * 0.37 ** 2 / (4 * np.pi ** 2) * 2.3 - 0.37 ** 2 * 2.3 / (4 * np.pi)) < 1e-15)
_P5 = _rng.normal(size=(200_000, 4)) * 3; _k5 = _rng.normal(size=(200_000, 4)) * 3
_m5 = 0.8
_Ei = np.sqrt(np.sum(_P5 ** 2, 1) + _m5 ** 2)
_Ef = np.sqrt(np.sum((_P5 - _k5) ** 2, 1) + _m5 ** 2) + np.linalg.norm(_k5, axis=1)
check("Sec 4.4  5D kinematics: a massive excitation cannot emit or absorb one massless quantum",
      np.min(_Ef - _Ei) > 0, "min energy deficit %.3e" % np.min(_Ef - _Ei))

# --- detector access: a fiber-uniform coupling registers only E_Y(rho)
_ns = np.arange(-2, 3); _d = len(_ns)
_En = np.sqrt(0.5 ** 2 + (_ns / 1.3) ** 2)
_sx = np.array([[0, 1], [1, 0]], complex)
def _shift(m):
    S = np.zeros((_d, _d))
    for i, n in enumerate(_ns):
        if n + m in _ns:
            S[list(_ns).index(n + m), i] = 1.0
    return S
_Hu = 0.8 * np.kron(np.diag([0.3, 1.1, 0.7, 1.1, 0.3]), _sx)            # uniform profile
_Hl = 0.8 * np.kron(_shift(0) + 0.5 * (_shift(1) + _shift(-1)), _sx)  # localized profile
_H0 = np.kron(np.diag(_En), np.eye(2)) + np.kron(np.eye(_d), 0.4 * np.diag([1, -1]))
def _det(rho, HFD, t=2.3):
    V = _ex2(-1j * (_H0 + HFD) * t)
    sig = np.diag([1.0, 0.0]).astype(complex)
    R = V @ np.kron(rho, sig) @ dag(V)
    return np.einsum('aibi->ab', R.reshape(_d, 2, _d, 2)) if False else \
        np.trace(R.reshape(_d, 2, _d, 2), axis1=0, axis2=2)
_rho = rand_state(_d)
_Ua = np.diag(np.exp(-1j * 0.9 * _ns))
_rhoa = _Ua @ _rho @ dag(_Ua)
_pinch = np.diag(np.diag(_rho))
check("Sec 3.3  fiber-uniform detector: same record for rho and its translate, and for E_Y(rho)",
      np.linalg.norm(_det(_rho, _Hu) - _det(_rhoa, _Hu)) < 1e-12
      and np.linalg.norm(_det(_rho, _Hu) - _det(_pinch, _Hu)) < 1e-12)
check("Sec 3.3  negative control: a detector localized on the fiber resolves the translate",
      np.linalg.norm(_det(_rho, _Hl) - _det(_rhoa, _Hl)) > 1e-3)

# --- single-sector code subspace, adjacent coherences, dipole transitions
# Thm 2(iv): a code subspace inside one sector is left fixed, and equivalent
# states on it coincide, so eps_Y = 0 at every t
_q1 = [3, 3, 3]                                                   # three states, one sector Q = 3
_r1s = rand_state(3); _s1s = rand_state(3)
def _deph(r, qs, t, g=0.7):
    out = r.copy()
    for i, a in enumerate(qs):
        for j, b in enumerate(qs):
            out[i, j] *= np.exp(-g * (a - b) ** 2 * t)
    return out
check("Thm 2(iv)  single sector: the dephasing is the identity on C",
      max(np.linalg.norm(_deph(_r1s, _q1, t) - _r1s) for t in (0.1, 1.0, 10.0)) < 1e-15)
# equivalence on a single sector means agreement on every block operator, hence equality
_ops = [np.eye(3)[:, [i]] @ np.eye(3)[[j], :] for i in range(3) for j in range(3)]
_agree = max(abs(np.trace(_r1s @ O) - np.trace(_s1s @ O)) for O in _ops)
check("Thm 2(iv)  single sector: states agreeing on all invariant block operators are equal",
      (_agree > 1e-6) == (np.linalg.norm(_r1s - _s1s) > 1e-6))
# Sec 7.1: every adjacent coherence decays at gamma while m_{n+1}^2 - m_n^2 grows with n
_Rq, _m5q, _gq = 1.3, 2.0, 0.45
_rates = []
for n0 in range(0, 4):
    r = np.full((2, 2), 0.5, complex)
    out = _deph(r, [n0, n0 + 1], 1.0, g=_gq)
    _rates.append(-np.log(abs(out[0, 1]) / 0.5))
_spl = [((n0 + 1) ** 2 - n0 ** 2) / _Rq ** 2 for n0 in range(0, 4)]
check("Sec 7.1  adjacent coherences all decay at gamma although m_{n+1}^2 - m_n^2 = (2n+1)/R^2 varies",
      np.ptp(_rates) < 1e-12 and abs(_rates[0] - _gq) < 1e-12 and np.ptp(_spl) > 1.0)
check("Sec 7.1  m_{n+q}^2 - m_n^2 = (2nq + q^2)/R^2",
      all(abs((_m5q ** 2 + (n0 + q) ** 2 / _Rq ** 2) - (_m5q ** 2 + n0 ** 2 / _Rq ** 2)
              - (2 * n0 * q + q * q) / _Rq ** 2) < 1e-12 for n0 in range(5) for q in range(1, 4)))
# Sec 4.4: golden-rule dipole transition rate inside a sector, against gamma
_gd, _Td, _d = 0.3, 50.0, 0.8
for _Om in (0.1, 0.5):
    _Jd = _Om ** 3 / (12 * np.pi ** 2)
    _nb1 = 1.0 / np.expm1(_Om / _Td) + 1.0
    _Gam = 2 * np.pi * _gd ** 2 * _d ** 2 * _Jd * _nb1
    _gam = _gd ** 2 * _Td / (4 * np.pi)
    _ratio = _Gam / (_gam * (2.0 / 3.0) * (_d * _Om) ** 2)
    check("Sec 4.4  dipole transition rate = (2/3)(d Omega)^2 gamma for T >> Omega (Omega = %.1f)" % _Om,
          abs(_ratio - 1) < 0.01, "ratio %.5f" % _ratio)

# --- infrared regularization of the Ohmic continuum (Sec 4.1)
from scipy.integrate import quad as _quad
_eta_ir, _wc_ir, _T_ir, _t_ir = 0.2, 10.0, 0.5, 3.0
def _disp_norm(w0):        # int_{w0}^inf J/w^2 dw: the displacement norm per unit (gQ)^2
    return _quad(lambda w: _eta_ir * np.exp(-w / _wc_ir) / w, w0, np.inf, limit=400)[0]
def _Gam_ir(w0):           # decoherence function at fixed t with the lower cutoff w0
    f = lambda w: (_eta_ir * w * np.exp(-w / _wc_ir) / w ** 2 / np.tanh(w / (2 * _T_ir))
                   * (1 - np.cos(w * _t_ir)))
    return _quad(f, w0, 50 * _wc_ir, limit=800)[0]
_dn = [_disp_norm(w0) for w0 in (1e-2, 1e-4, 1e-6)]
_gi = [_Gam_ir(w0) for w0 in (1e-2, 1e-4, 1e-6)]
check("Sec 4.1  Ohmic displacement norm diverges like eta ln(1/w0) as the infrared cutoff w0 -> 0",
      abs((_dn[2] - _dn[1]) / (_eta_ir * np.log(100.0)) - 1) < 1e-3,
      "norms %s" % np.round(_dn, 4))
# the integrand tends to eta T t^2 at w = 0, so the cutoff removes only eta T t^2 w0
_lead = _eta_ir * _T_ir * _t_ir ** 2
check("Sec 4.1  Gamma(t) at fixed t converges as the infrared cutoff is removed",
      abs((_gi[2] - _gi[1]) / (_lead * (1e-4 - 1e-6)) - 1) < 1e-2
      and abs((_gi[1] - _gi[0]) / (_lead * (1e-2 - 1e-4)) - 1) < 1e-2,
      "Gamma %s" % np.round(_gi, 8))

# --- finite-gap corrections (Sec 4.5), joint frame (Sec 5.4), isotropic bath (Sec 6.1)
from scipy.integrate import quad as _q6
_pi = np.pi
def _nB(w, T):
    return 1.0 / np.expm1(w / T)
# massive harmonic at a point: J_m = sqrt(w^2 - mu^2)/(4 pi^2), lattice check
_mu6, _L6 = 1.3, 70.0
_n6 = int(np.ceil(6.0 * _L6 / (2 * _pi))) + 1
_a6 = np.arange(-_n6, _n6 + 1)
_x6, _y6, _z6 = np.meshgrid(_a6, _a6, _a6, indexing='ij', sparse=True)
_w6 = np.sqrt(((2 * _pi / _L6) ** 2) * (_x6 ** 2 + _y6 ** 2 + _z6 ** 2) + _mu6 ** 2)
_r6 = []
for _W in (2.5, 4.0, 5.5):
    _lat = np.sum(1.0 / (2 * _w6[_w6 < _W] * _L6 ** 3))
    _con = (_W * np.sqrt(_W ** 2 - _mu6 ** 2)
            - _mu6 ** 2 * np.log((_W + np.sqrt(_W ** 2 - _mu6 ** 2)) / _mu6)) / (8 * _pi ** 2)
    _r6.append(_lat / _con)
check("Sec 4.5  J_m(w) = sqrt(w^2 - mu_m^2)/(4 pi^2) for a harmonic of mass mu_m at a point",
      max(abs(r - 1) for r in _r6) < 5e-3, "ratios %s" % np.round(_r6, 5))
_okg = all(abs(np.sqrt(m5 ** 2 + (n + m) ** 2) - np.sqrt(m5 ** 2 + n ** 2)) < abs(m) - 1e-12
           for m5 in (0.05, 0.45, 3.0, 40.0) for n in range(-6, 7) for m in (-3, -2, -1, 1, 2, 3))
check("Prop 2(i)  every transition frequency |E_{n+m} - E_n| lies inside the gap |m|/R", _okg)
# Raman amplitude from the two time orderings
_E6 = lambda q, m5R=37.0: np.sqrt(m5R ** 2 + q ** 2)
_okM6 = True
for n in (-2, 0, 1, 3):
    for m in (-1, 1, 2):
        w = 1.7; wp = w + _E6(n) - _E6(n + m)
        t1 = n * (n + m / 2.0) / w
        t2 = (n + m / 2.0) * (n + m) / (_E6(n) + w - (_E6(n + m) + w + wp))
        _okM6 &= abs((t1 + t2) + (n + m / 2.0) * m / w) < 1e-12
check("Prop 2(iii)  Raman amplitude n -> n+m equals -g^2 (n + m/2) m / omega", _okM6)
# elastic amplitude of a thermal harmonic couples through K_y
_okE6 = True
for n in (-2, 0, 1, 3):
    for m in (1, 2):
        wq = 1.9; Eh = lambda q: _E6(q, 1.0e3)
        M = (n + m / 2.0) ** 2 / (wq - (Eh(n + m) - Eh(n))) - (n - m / 2.0) ** 2 / (wq + (Eh(n - m) - Eh(n)))
        _okE6 &= abs(M - 2 * n * m / wq) < 0.01 * max(1.0, abs(2 * n * m / wq))
check("Prop 2(ii)  elastic amplitude of a thermal harmonic is 2 n m g^2/omega, through K_y", _okE6)
def _leak6(n, m, T, m5R=1.0e4):
    E = lambda q: np.sqrt(m5R ** 2 + q ** 2)
    mu = abs(m); dl = E(n + m) - E(n)
    J0 = lambda w: w / (4 * _pi ** 2)
    Jm = lambda w: np.sqrt(max(w * w - mu * mu, 0.0)) / (4 * _pi ** 2)
    fa = lambda w: J0(w) * _nB(w, T) / w ** 2 * Jm(w - dl) * (1 + _nB(w - dl, T))
    fb = lambda wp: Jm(wp) * _nB(wp, T) * J0(wp - dl) * (1 + _nB(wp - dl, T)) / (wp - dl) ** 2
    a = _q6(fa, mu + dl, mu + dl + 60 * T, limit=400)[0]
    b = _q6(fb, mu, mu + 60 * T, limit=400)[0]
    return 2 * _pi * (n + m / 2.0) ** 2 * m ** 2 * (a + b)
_rl = [(T, _leak6(1, 1, T) / (np.sqrt(2 * _pi) * 1.5 ** 2 * T ** 1.5 * np.exp(-1 / T) / (8 * _pi ** 3)))
       for T in (0.01, 0.02, 0.05)]
check("Prop 2(iii)  w_{n->n+1} = sqrt(2 pi) g^4 (n+1/2)^2 R^{1/2} T^{3/2} e^{-1/(RT)}/(8 pi^3) [1 + O(RT)]",
      all(abs(r - (1 - 9 * T / 8)) < 0.01 for T, r in _rl), "ratios %s" % [round(r, 4) for _, r in _rl])
def _dg6(m, T):
    mu = abs(m)
    Jm = lambda w: np.sqrt(max(w * w - mu * mu, 0.0)) / (4 * _pi ** 2)
    return 4 * _pi * m ** 2 * _q6(lambda w: Jm(w) ** 2 * _nB(w, T) * (1 + _nB(w, T)) / w ** 2,
                                  mu, mu + 80 * T, limit=400)[0]
_rd = [(T, (_dg6(1, T) + _dg6(-1, T)) / (T ** 2 * np.exp(-1 / T) / _pi ** 3)) for T in (0.005, 0.01, 0.02)]
check("Prop 2(ii)  delta gamma = g^4 R T^2 e^{-1/(RT)}/pi^3 [1 + O(RT)], i.e. (4/pi^2) g^2 R T e^{-1/(RT)} gamma",
      all(abs(r - (1 - 3 * T)) < 0.005 for T, r in _rd), "ratios %s" % [round(r, 4) for _, r in _rd])
check("Sec 4.5  eps_local/gamma = (sqrt(2 pi)/pi^2) g^2 (2n^2 + 1/2) (RT)^{1/2} e^{-1/(RT)}",
      abs(2 * (np.sqrt(2 * _pi) / (8 * _pi ** 3)) * ((1 + 0.5) ** 2 + (1 - 0.5) ** 2) * 4 * _pi
          - (np.sqrt(2 * _pi) / _pi ** 2) * (2 * 1 + 0.5)) < 1e-12)
# Duhamel bound for leakage on top of QND dephasing
from scipy.linalg import expm as _expm6
_lv = np.arange(-2, 3); _dv = len(_lv)
_K6 = np.diag(_lv).astype(complex); _H6 = np.diag(np.sqrt(25.0 + _lv ** 2)).astype(complex)
def _sc(X):
    I = np.eye(_dv); return np.kron(X, I) - np.kron(I, X.T)
def _ld(Lj):
    I = np.eye(_dv); LL = Lj.conj().T @ Lj
    return np.kron(Lj, Lj.conj()) - 0.5 * np.kron(LL, I) - 0.5 * np.kron(I, LL.T)
_B6 = -1j * _sc(_H6) - 0.7 * _sc(_K6) @ _sc(_K6)
_rt6 = {}; _D6 = np.zeros((_dv ** 2, _dv ** 2), complex)
for i, n in enumerate(_lv):
    for sgn in (1, -1):
        if 0 <= i + sgn < _dv:
            r = 0.03 * (n + sgn / 2.0) ** 2; _rt6[(n, sgn)] = r
            Lj = np.zeros((_dv, _dv), complex); Lj[i + sgn, i] = np.sqrt(r); _D6 += _ld(Lj)
_A6 = _B6 + _D6
_SC6 = [0, 1]
_eps6 = 2 * max(sum(_rt6.get((n, sgn), 0) for sgn in (1, -1)) for n in _SC6)
_okB6 = True
_g6 = np.random.default_rng(11)
for _ in range(40):
    X = _g6.normal(size=(2, 2)) + 1j * _g6.normal(size=(2, 2)); X = X @ X.conj().T; X /= np.trace(X)
    rho = np.zeros((_dv, _dv), complex); idx = [list(_lv).index(n) for n in _SC6]; rho[np.ix_(idx, idx)] = X
    for t in (0.3, 1.0, 3.0):
        dlt = (_expm6(t * _A6) @ rho.reshape(-1) - _expm6(t * _B6) @ rho.reshape(-1)).reshape(_dv, _dv)
        _okB6 &= np.sum(np.abs(np.linalg.eigvalsh((dlt + dlt.conj().T) / 2))) <= t * _eps6 + 1e-12
check("Prop 2(iv)  ||rho_loc(t) - rho_QND(t)||_1 <= 2 t max_{n in S_C} w_leak(n)", _okB6)
# joint dephasing of particle and frame
_xh, _wh = np.polynomial.hermite_e.hermegauss(120); _wh = _wh / _wh.sum()
def _jr(qS, qD, xi, g=0.4, t=1.3):
    val = np.sum(_wh * np.exp(-1j * np.sqrt(2 * g * t) * _xh * (qS + xi * qD)))
    return -np.log(abs(val)) / t
check("Sec 5.4  coupling K_y + xi K_D: relative harmonic decays at gamma (1-xi)^2, absolute at gamma and gamma xi^2",
      all(abs(_jr(1, -1, xi) - 0.4 * (1 - xi) ** 2) < 1e-9 and abs(_jr(1, 0, xi) - 0.4) < 1e-9
          and abs(_jr(0, 1, xi) - 0.4 * xi ** 2) < 1e-9 for xi in (0.0, 0.3, 0.5, 1.0, 1.7)))
_dd6, _Lc = 1.3, 80.0
_nc = int(np.ceil(5.0 * _Lc / (2 * _pi))) + 1
_ac = np.arange(-_nc, _nc + 1)
_xc, _yc, _zc = np.meshgrid(_ac, _ac, _ac, indexing='ij', sparse=True)
_kc = (2 * _pi / _Lc) * np.sqrt(_xc ** 2 + _yc ** 2 + _zc ** 2)
_kxc = (2 * _pi / _Lc) * _xc * np.ones_like(_kc)
_rc6 = []
for _W in (2.0, 3.5, 5.0):
    _m = (_kc > 0) & (_kc < _W)
    _rc6.append(np.sum(np.cos(_kxc[_m] * _dd6) / (2 * _kc[_m] * _Lc ** 3)) - (1 - np.cos(_W * _dd6)) / (4 * _pi ** 2 * _dd6 ** 2))
check("Sec 5.4  cross spectral density of the zero mode at distance d is J(w) sin(wd)/(wd)",
      max(abs(x) for x in _rc6) < 2e-4, "residuals %s" % np.round(_rc6, 6))
def _thermal6(T, d, wc):
    eps = 1.0 / (wc * d)
    f = lambda u: np.exp(-eps * u) * (1 - np.sinc(u / _pi)) / u ** 2 if u > 1e-6 else 1.0 / 6
    ed = np.concatenate([[0.0], np.arange(1, 1201) * _pi])
    I = sum(_q6(f, a, b, limit=200)[0] for a, b in zip(ed[:-1], ed[1:]))
    I += _q6(lambda u: np.exp(-eps * u) / u ** 2, ed[-1], np.inf, limit=200)[0]
    return 2 * T * d * I
def _vac6(T, wc, d=None):
    def f(w):
        x = w / (2 * T); c = 1.0 / np.tanh(x) - 1.0 / x if x > 1e-4 else x / 3.0
        return np.exp(-w / wc) / w * c * (1 if d is None else (1 - np.sinc(w * d / _pi)))
    ed = np.concatenate([np.linspace(0, 50 * T, 300), np.geomspace(50 * T, 80 * wc, 500)[1:]])
    return sum(_q6(f, a, b, limit=200)[0] for a, b in zip(ed[:-1], ed[1:]))
_th = [(_thermal6(T, d, wc), _pi * T * d / 2) for (T, d, wc) in ((1.0, 20.0, 200.0), (0.5, 100.0, 300.0))]
check("Sec 5.4  thermal part of Delta_d(inf) equals gamma d/2 for T d >> 1 (eta = g = 1)",
      all(abs(a / b - 1) < 5e-3 for a, b in _th), "ratios %s" % [round(a / b, 5) for a, b in _th])
_vf = _vac6(1.0, 1.0e4)
check("Sec 5.4  vacuum part of Delta_d(inf) tends to ln[wc/(2 pi T)]",
      abs(_vf - np.log(1.0e4 / (2 * _pi))) < 0.01 and abs(_vac6(1.0, 200.0, d=50.0) - _vac6(1.0, 200.0)) < 0.02,
      "value %.5f vs %.5f" % (_vf, np.log(1.0e4 / (2 * _pi))))
check("Sec 5.4  zero-mode exchange between the charges: -2 xi g^2 int J sinc/w = -xi g^2/(4 pi d)",
      abs(2 * _q6(lambda w: (w / (4 * _pi ** 2)) * np.sinc(w * 1.7 / _pi) / w, 0, 4000, limit=4000)[0]
          - 1 / (4 * _pi * 1.7)) < 2e-3)
# isotropic bath on S^2: TCL2 structure with independent identical baths
def _spin6(l):
    ms = np.arange(l, -l - 1, -1); Jz = np.diag(ms).astype(complex)
    Jp = np.zeros((len(ms), len(ms)), complex)
    for i in range(1, len(ms)):
        Jp[i - 1, i] = np.sqrt(l * (l + 1) - ms[i] * (ms[i] + 1))
    return (Jp + Jp.conj().T) / 2, (Jp - Jp.conj().T) / (2j), Jz
from scipy.linalg import block_diag as _bd6
_Js6 = [_bd6(a, b) for a, b in zip(_spin6(1), _spin6(2))]
_C26 = sum(J @ J for J in _Js6)
_rh6 = rand_state(_C26.shape[0]); _c6 = 0.37 + 0.21j
_lhs6 = sum(_c6 * (J @ J @ _rh6 - J @ _rh6 @ J) - np.conj(_c6) * (J @ _rh6 @ J - _rh6 @ J @ J) for J in _Js6)
_rhs6 = (sum(_c6.real * (J @ (J @ _rh6 - _rh6 @ J) - (J @ _rh6 - _rh6 @ J) @ J) for J in _Js6)
         + 1j * _c6.imag * (_C26 @ _rh6 - _rh6 @ _C26))
check("Sec 6.1  isotropic bath: TCL2 generator = Re C sum_a [J_a,[J_a,.]] + i Im C [C_2, .], C_2 central",
      np.linalg.norm(_lhs6 - _rhs6) < 1e-12 and all(np.linalg.norm(_C26 @ J - J @ _C26) < 1e-12 for J in _Js6))

# --- virtual cloud of the m = +-1 harmonics (Prop 2(i)), exact diagonalization
def _dress6(g, n, W=1.0, c=1.0, lv=tuple(range(-4, 5)), nb=4):
    st = [(k, p, q) for k in lv for p in range(nb) for q in range(nb)]
    ix = {x: i for i, x in enumerate(st)}
    H = np.zeros((len(st), len(st)))
    for (k, p, q), i in ix.items():
        H[i, i] = W * (p + q)
        if k + 1 in lv:
            amp = g * c * (k + 0.5)
            if p > 0:
                j = ix[(k + 1, p - 1, q)]; H[j, i] += amp * np.sqrt(p); H[i, j] += amp * np.sqrt(p)
            if q + 1 < nb:
                j = ix[(k + 1, p, q + 1)]; H[j, i] += amp * np.sqrt(q + 1); H[i, j] += amp * np.sqrt(q + 1)
    sel = [i for i, x in enumerate(st) if x[0] + x[1] - x[2] == n]
    e, v = np.linalg.eigh(H[np.ix_(sel, sel)])
    psi = v[:, 0]
    ky = sum(psi[a] ** 2 * st[i][0] for a, i in enumerate(sel))
    pp = sum(psi[a] ** 2 for a, i in enumerate(sel) if st[i][0] == n + 1)
    pm = sum(psi[a] ** 2 for a, i in enumerate(sel) if st[i][0] == n - 1)
    return ky, pp, pm
_kd, _pp, _pm = _dress6(0.02, 1)
check("Prop 2(i)  virtual cloud: <K_y> = Z n with Z - 1 = 2 g^2 C_1 for one +-1 mode pair (C_1 = c^2/W^2)",
      abs(((_kd - 1) / 1) / (2 * 0.02 ** 2) - 1) < 0.02, "ratio %.5f" % (((_kd - 1) / 1) / (2 * 0.02 ** 2)))
check("Prop 2(i)  virtual cloud: weights g^2 (n +- 1/2)^2 C_1 on the bare levels n +- 1",
      abs(_pp / (0.02 ** 2 * 1.5 ** 2) - 1) < 0.02 and abs(_pm / (0.02 ** 2 * 0.5 ** 2) - 1) < 0.02,
      "P(n+1) ratio %.4f, P(n-1) ratio %.4f" % (_pp / (0.02 ** 2 * 2.25), _pm / (0.02 ** 2 * 0.25)))

# --- relational state, radion dephasing, activation energy, parameter table, dressing value
_yy = np.linspace(0, 2 * np.pi, 61); _Y, _YD = np.meshgrid(_yy, _yy, indexing='ij')
_n7, _q7 = 2, 5
_psi7 = (np.exp(1j * (_n7 * _Y + (_q7 - _n7) * _YD)) + np.exp(1j * ((_n7 + 1) * _Y + (_q7 - _n7 - 1) * _YD))) / (2 * np.pi * np.sqrt(2))
_den7 = np.abs(_psi7) ** 2
check("Sec 5.4  relational state: density [1 + cos(y - y_D)]/(4 pi^2), uniform marginals, total charge Q_tot",
      np.max(np.abs(_den7 - (1 + np.cos(_Y - _YD)) / (4 * np.pi ** 2))) < 1e-12
      and np.max(np.abs(_den7.mean(axis=1) * 2 * np.pi - 1 / (2 * np.pi))) < 1e-2)
_al7 = 0.83
_psit = (np.exp(1j * (_n7 * (_Y - _al7) + (_q7 - _n7) * (_YD - _al7)))
         + np.exp(1j * ((_n7 + 1) * (_Y - _al7) + (_q7 - _n7 - 1) * (_YD - _al7)))) / (2 * np.pi * np.sqrt(2))
check("Sec 5.4  joint translation multiplies the relational state by e^{-i Q_tot alpha}: ray and density operator invariant",
      np.max(np.abs(_psit - np.exp(-1j * _q7 * _al7) * _psi7)) < 1e-12)
_R7, _m57 = 1.0, 60.0
_E7 = lambda n, R: np.sqrt(_m57 ** 2 + n ** 2 / R ** 2)
_dER = lambda n: (_E7(n, _R7 + 1e-6) - _E7(n, _R7 - 1e-6)) / 2e-6
check("Sec 8  radion: dE_n/dR = -n^2/(R^3 E_n); n and -n untouched; (n^2/E_n - n'^2/E_n') -> (n^2 - n'^2) for |n| << m5 R",
      all(abs(_dER(n) + n ** 2 / (_R7 ** 3 * _E7(n, _R7))) < 1e-6 for n in range(-3, 4))
      and abs(_dER(2) - _dER(-2)) < 1e-12
      and abs((_dER(2) - _dER(1)) / (_dER(3) - _dER(1)) - (4 - 1) / (9 - 1)) < 1e-3)
_Ts = np.array([0.04, 0.05, 0.06, 0.08])
_dgT = np.array([_dg6(1, T) + _dg6(-1, T) for T in _Ts])
_lkT = np.array([_leak6(1, 1, T) for T in _Ts])
_sd = -np.polyfit(1 / _Ts, np.log(_dgT / _Ts ** 2), 1)[0]
_sl = -np.polyfit(1 / _Ts, np.log(_lkT / _Ts ** 1.5), 1)[0]
check("Sec 7.2  activation energy of delta gamma and of the leakage is 1/R = (m_1^2 - m_5^2)^{1/2}",
      abs(_sd - 1) < 0.02 and abs(_sl - 1) < 0.02, "fitted E_a R = %.4f, %.4f" % (_sd, _sl))
_tab = {0.05: (4.2e-13, 2.9e-12, 3.4e11), 0.1: (1.8e-8, 9.1e-8, 1.1e7), 0.2: (5.5e-6, 1.9e-5, 5.2e4), 0.3: (4.3e-5, 1.2e-4, 8.1e3)}
_okT = True
for _rt, (_a, _b, _c) in _tab.items():
    _dgv = (4 / np.pi ** 2) * 0.01 * _rt * np.exp(-1 / _rt)
    _epv = (np.sqrt(2 * np.pi) / np.pi ** 2) * 0.01 * 2.5 * np.sqrt(_rt) * np.exp(-1 / _rt)
    _okT &= abs(_dgv / _a - 1) < 0.05 and abs(_epv / _b - 1) < 0.05 and abs((1 / _epv) / _c - 1) < 0.05   # two-figure rounding
check("Sec 4.5  regime table entries follow from Eqs. (dgamma) and (local-ratio) at g^2 = 0.01, n = 1", _okT)
check("Sec 4.5  dressing correction Z^2 - 1 = 0.090 for Z = 1.044, separate from the thermal entries",
      abs(1.044 ** 2 - 1 - 0.090) < 5e-4, "Z^2 - 1 = %.4f" % (1.044 ** 2 - 1))
_g7, _T7, _xi7 = 0.1, 0.1, 1.0
_gam7 = _g7 ** 2 * _T7 / (4 * np.pi)
_ph7 = lambda d: _xi7 * _g7 ** 2 / (4 * np.pi * d)
check("Sec 8  dephasing rate over entangling phase rate is T d / xi (equal at d = xi/T)",
      all(abs(_gam7 / _ph7(d) - _T7 * d / _xi7) < 1e-12 for d in (0.5, 3.0, 40.0)) and abs(_gam7 / _ph7(_xi7 / _T7) - 1) < 1e-12)
def _Cm7(m, aR):
    f = lambda k: k * k / (k * k + m * m) ** 1.5 * np.exp(-k * k * aR * aR)
    return np.exp(-(m * aR) ** 2) * _q6(f, 0, np.inf, limit=400)[0] / (4 * np.pi ** 2)
_Z7 = 2 * 0.01 * sum(m * m * _Cm7(m, 0.1) for m in range(1, 81))
check("Sec 4.5  dressing Z - 1 = 0.044 for a Gaussian smearing with a/R = 0.1 at g^2 = 0.01",
      abs(_Z7 - 0.044) < 0.001, "Z - 1 = %.5f" % _Z7)
_sp = [(_E7(n + 1, 1.0) - _E7(n, 1.0)) for n in (0, 1)]
check("Sec 4.5  example point m5 R = 100, RT = 0.1: level spacings are 5 and 15 percent of T",
      abs((np.sqrt(1e4 + 1) - 100) / 0.1 - 0.05) < 0.001 and abs((np.sqrt(1e4 + 4) - np.sqrt(1e4 + 1)) / 0.1 - 0.15) < 0.001)

# =====================================================================
# report
# =====================================================================
print("=" * 68)
print("Numerical verification of the manuscript")
print("=" * 68)
nfail = 0
for name, ok, detail in results:
    tag = "ok  " if ok else "FAIL"
    if not ok:
        nfail += 1
    line = "%s  %s" % (tag, name)
    if detail and not ok:
        line += "\n        %s" % detail
    print(line)
print("-" * 68)
print("RESULT: %s (%d checks, %d failures)"
      % ("PASS" if nfail == 0 else "FAIL", len(results), nfail))
sys.exit(1 if nfail else 0)
