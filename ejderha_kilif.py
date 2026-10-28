"""AirPods 4 kılıfı (v2): menteşeli kapak, çanta halkası deliği, kılıfı saran büyük ejderha.

Kullanım:  python ejderha_kilif.py          (hızlı deneme için:  H=0.4 python ejderha_kilif.py)
Ölçüler mm. Kılıf oturmazsa önce ÖLÇÜLER, BOŞLUK ve MENTEŞE değerlerini düzelt.

Kapak gövdeye iki yandan pimle bağlanır: kapağın yan plakasındaki pim, gövdenin kulağındaki
deliğe oturur. Pim ekseni AirPods kutusunun kendi menteşe eksenine denk gelecek yere konur,
böylece kutu açılıp kapanırken kılıf zorlanmaz.
"""
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from skimage.measure import marching_cubes
import trimesh
import manifold3d as mf

# ---------------- ÖLÇÜLER (AirPods 4 şarj kutusu, Apple: 46,2 × 50,1 × 21,2 mm) ----------------
GEN, YUK, DER = 50.1, 46.2, 21.2   # genişlik (x), yükseklik (z), derinlik (y); ön = -y
R_ON = 14.0      # önden bakınca köşe yuvarlaklığı
R_KENAR = 7.0    # yan kenarların yuvarlaklığı (derinlik yönünde)
EK_YERI = 32.0   # kapak çizgisinin alttan yüksekliği (kendi kutunda ölç!)
LED_Z = 26.0     # ön ışık deliğinin alttan yüksekliği

# ---------------- BASKI ----------------
BOSLUK = 0.20        # kutu ile gövde parçası arası (PLA: 0.20, TPU: 0.05)
BOSLUK_KAPAK = 0.15  # kapakta biraz daha sıkı
KALINLIK = 1.8       # kılıf duvarı
ARALIK = 0.35        # gövde-kapak arası boşluğun yarısı
H = float(os.environ.get("H", 0.2))   # çözünürlük (küçük = daha detaylı, daha yavaş)

# ---------------- MENTEŞE ----------------
MENTESE_ICERI = 0.3  # kutunun menteşe ekseninin arka yüzeyden içeri uzaklığı (tahmin)
MENTESE_ALTI = 0.5   # eksenin kapak çizgisinden aşağı uzaklığı (tahmin)
ACILMA = 100.0       # kapağın en fazla açılma açısı (derece)
PLAKA = 1.8          # pim plakalarının kalınlığı
R_IC = 4.2           # kapak plakasının yarıçapı
R_KULAK = 6.6        # gövde kulağının yarıçapı
PIM_R = 1.5          # pim yarıçapı
DELIK_R = 1.85       # kulaktaki delik (pimden biraz bol: eksen tahmini tutmazsa tolerans)
YUVA = 2.75          # pimin kulağa girdiği ağız genişliği (pimden dar: tık diye oturur)

# ---------------- ÇANTA HALKASI ----------------
HALKA_Y, HALKA_Z = -3.0, 23.0   # sağ yanda, kapak çizgisinin altında
HALKA_DELIK = 2.8               # delik yarıçapı (5,6 mm: anahtarlık halkası, küçük karabina)
HALKA_DIS = 5.2                 # kulp dış yarıçapı
HALKA_KAL = 5.0                 # kulp kalınlığı

AX, AY, AZ = GEN / 2, DER / 2, YUK / 2
DIS = BOSLUK + KALINLIK
EK = EK_YERI
MY, MZ = AY - MENTESE_ICERI, EK - MENTESE_ALTI
X0 = AX + DIS + 0.4          # kapak plakasının iç yüzü
X1 = X0 + PLAKA + 0.35       # gövde kulağının iç yüzü
X2 = X1 + PLAKA              # gövde kulağının dış yüzü


def kutu_sdf(P):
    """Kutunun yüzeyine işaretli uzaklık (içeride negatif)."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    rr = R_ON - R_KENAR
    qx = np.abs(x) - (AX - R_KENAR) + rr
    qz = np.abs(z - AZ) - (AZ - R_KENAR) + rr
    d2 = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - rr
    qy = np.abs(y) - (AY - R_KENAR)
    return np.hypot(np.maximum(d2, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(d2, qy), 0) - R_KENAR


def normal(P, e=0.01):
    g = np.stack([kutu_sdf(P + np.eye(3)[i] * e) - kutu_sdf(P - np.eye(3)[i] * e) for i in range(3)], -1)
    return g / np.linalg.norm(g, axis=-1, keepdims=True)


def yuzeye(q, kalk):
    """Noktayı kılıfın dış yüzeyine indirip `kalk` kadar dışarı çıkarır."""
    for _ in range(4):
        q = q - normal(q) * (kutu_sdf(q) - DIS - kalk)[..., None]
    return q


def nrm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def kapsul(P, a, b, ra, rb):
    ab = b - a
    h = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - a - h[:, None] * ab, axis=1) - (ra + (rb - ra) * h)


def elipsoid(P, c, eks, r):
    q = ((P - c) @ eks.T) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def ray(a, z):
    """Kutunun etrafında `a` açısında (0 = ön, π/2 = sağ, π = arka), `z` yüksekliğinde dış yüzey noktası."""
    yon = np.array([np.sin(a), -np.cos(a), 0.0])
    lo, hi = 0.0, 45.0
    for _ in range(40):
        m = (lo + hi) / 2
        if kutu_sdf(yon * m + [0, 0, z]) < DIS:
            lo = m
        else:
            hi = m
    return yon * lo + [0, 0, z]


# ---------------- KAFA (kapağın üstünde, sağ omuzdan aşağı bakıyor) ----------------
KB = 2.2                                    # kafa büyüklüğü
K = np.array([11.0, -2.5, 50.0])            # kafatası merkezi
f = nrm(np.array([1.0, -0.5, -0.28]))      # bakış yönü
u = nrm(np.array([0, 0, 1.0]) - f[2] * f)
sh = np.cross(f, u)


def kafa(P):
    return KB * _kafa(K + (P - K) / KB)


def _kafa(P):
    def nk(a, b, c):
        return K + f * a + u * b + sh * c

    d = elipsoid(P, K, np.stack([f, sh, u]), np.array([4.0, 3.5, 2.8]))                     # kafatası
    d = smin(d, kapsul(P, nk(1.5, 0.5, 0), nk(8.2, 0.3, 0), 2.4, 1.35), 1.0)               # üst çene
    d = smin(d, kapsul(P, nk(5.5, 1.3, 0), nk(0.5, 2.6, 0), 0.5, 0.7), 0.5)                # burun sırtı
    d = smin(d, kapsul(P, nk(0.2, -1.0, 0), nk(7.0, -2.9, 0), 1.9, 0.95), 0.6)             # alt çene (açık)
    d = smin(d, kapsul(P, nk(6.6, -3.0, 0), nk(7.6, -3.9, 0), 0.6, 0.15), 0.4)             # çene dikeni
    d = np.maximum(d, -kapsul(P, nk(2.6, -0.75, 0), nk(9.5, -1.5, 0), 0.5, 1.15))          # ağız
    for k in (1, -1):
        d = smin(d, kapsul(P, nk(3.0, 1.9, k * 1.7), nk(-0.4, 2.4, k * 2.8), 0.6, 0.85), 0.4)  # kaş
        g = nk(1.7, 1.15, k * 2.5)
        d = np.maximum(d, -(np.linalg.norm(P - g, axis=1) - 1.05))                         # göz çukuru
        d = np.minimum(d, np.linalg.norm(P - (g + nrm(K - g) * 0.3), axis=1) - 0.9)         # göz (çukurun dibine oturur)
        # büyük boynuzlar: geriye, sonra yukarı kıvrılır
        m = [nk(-1.4, 1.8, k * 1.8), nk(-5.8, 2.2, k * 3.1), nk(-9.8, 2.6, k * 3.6), nk(-12.6, 4.4, k * 3.3)]
        d = smin(d, np.minimum.reduce([kapsul(P, m[0], m[1], 1.3, 0.9), kapsul(P, m[1], m[2], 0.9, 0.5),
                                       kapsul(P, m[2], m[3], 0.5, 0.08)]), 0.5)
        d = smin(d, kapsul(P, nk(-2.0, 0.3, k * 2.9), nk(-6.0, -0.6, k * 4.4), 0.7, 0.1), 0.4)  # küçük boynuz
        for j, (a, b) in enumerate([(0.6, 0.8), (-0.4, -0.6), (-1.6, -2.0)]):              # yanak dikenleri
            d = smin(d, kapsul(P, nk(-0.5 - 0.3 * j, a * 0.6, k * 3.0), nk(-3.6 - 0.4 * j, b, k * 4.6), 0.45, 0.06), 0.3)
        for x in (3.6, 4.9, 6.2, 7.4):                                                      # üst dişler
            r0 = 0.42 if x == 7.4 else 0.3
            d = smin(d, kapsul(P, nk(x, -0.6, k * 1.15), nk(x + 0.15, -1.75 if x == 7.4 else -1.35, k * 1.05), r0, 0.04), 0.15)
        for x in (3.2, 4.6, 5.9):                                                           # alt dişler
            d = smin(d, kapsul(P, nk(x, -1.9 - 0.27 * x, k * 0.95), nk(x + 0.1, -1.05 - 0.25 * x, k * 0.9), 0.28, 0.04), 0.15)
        d = np.maximum(d, -(np.linalg.norm(P - nk(7.9, 1.0, k * 0.75), axis=1) - 0.45))   # burun deliği
    return d


# ---------------- EJDERHANIN OMURGASI ----------------
# Kafanın arkasından başlar, kapağın üstünden sola geçer, önden kapak çizgisini keser,
# gövdeyi bir tur dolaşır, kuyruk önde altta biter. Arka menteşe bölgesine ve sağdaki kulpa uğramaz.
kontrol = [K - f * 3.0 * KB, np.array([-2.0, -1.0, 49.0]), np.array([-13.0, -3.5, 47.5]),
           ray(-1.05, 41.0), ray(-0.78, 33.0), ray(-0.42, 24.5), ray(-0.05, 18.0), ray(0.45, 13.5),
           ray(np.pi / 2, 10.5), ray(2.3, 9.0), ray(np.pi, 8.5), ray(3.95, 9.0), ray(3 * np.pi / 2, 11.0),
           ray(2 * np.pi - 0.75, 8.5), ray(2 * np.pi - 0.2, 5.0), ray(2 * np.pi + 0.3, 4.0), ray(2 * np.pi + 0.62, 4.6)]
kontrol = np.array(kontrol)
uz = np.r_[0, np.cumsum(np.linalg.norm(np.diff(kontrol, axis=0), axis=1))]
N = 4000
s = np.linspace(0, 1, N)
q = CubicSpline(uz / uz[-1], kontrol)(s)
r = np.interp(s, [0, 0.05, 0.14, 0.3, 0.55, 0.8, 1.0], [3.3, 3.7, 4.3, 4.8, 4.5, 2.8, 0.9])
kalk = np.interp(s, [0, 0.03, 0.06], [2.6, 1.0, 0.0])      # boyun kafaya doğru yükselir
yuz = yuzeye(q, 0.0)
n = normal(yuz)
C = yuz + n * (0.32 * r + kalk)[:, None]                   # gövde yarı gömülü
t = nrm(np.gradient(C, axis=0))
yan = nrm(np.cross(n, t))
n = np.cross(t, yan)
yay = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]

# ---------------- PARÇALAR ----------------
parcalar = [(K, 24.0 * KB, kafa)]   # (merkez, yarıçap, sdf) — sadece yakın noktalarda hesaplanır


def bacak_ekle(i, ileri):
    for k in (1, -1):
        c, ni, ti, si = C[i], n[i], t[i], yan[i] * k
        omuz = c + si * r[i] * 0.4
        diz = c + si * (r[i] + 3.2) + ni * 2.6 + ti * ileri * 1.5
        ayak = yuzeye(c + si * (r[i] + 6.2) + ti * ileri * 3.0, 1.1)
        parmak, pence = [], []
        for a in (-0.8, -0.27, 0.27, 0.8):
            yon = nrm(si + ti * a)
            p = yuzeye(ayak + yon * 3.6, 0.65)
            parmak.append(p)
            pence.append(yuzeye(p + yon * 1.9, -0.3))

        def bacak(P, omuz=omuz, diz=diz, ayak=ayak, parmak=parmak, pence=pence, rr=r[i]):
            d = smin(kapsul(P, omuz, diz, rr * 0.55, 1.8), kapsul(P, diz, ayak, 1.8, 1.35), 0.8)
            for p, pc in zip(parmak, pence):
                d = smin(d, kapsul(P, ayak, p, 0.9, 0.55), 0.4)
                d = smin(d, kapsul(P, p, pc, 0.55, 0.06), 0.15)
            return d

        parcalar.append((ayak, 14.0, bacak))


bacak_ekle(int(0.095 * N), -1)   # ön ayaklar: kapağın sol omzunda
bacak_ekle(int(0.535 * N), 1)    # arka ayaklar: sırtta

# kuyruk ucu (mızrak)
ce, te, ne_, se = C[-1] + t[-1] * 2.4, t[-1], n[-1], yan[-1]


def mizrak(P):
    w = P - ce
    a, b = 4.6, 3.2
    q = np.abs(w @ te) * b + np.abs(w @ se) * a - a * b
    return np.maximum(q / np.hypot(a, b), np.abs(w @ ne_) - 1.0)


parcalar.append((ce, 9.0, mizrak))

# sırt dikenleri (geriye yatık)
ds = []
for L in np.arange(yay[-1] * 0.05, yay[-1] * 0.97, 2.9):
    i = np.searchsorted(yay, L)
    o = r[i] / 4.8
    ds.append((C[i] + n[i] * (0.85 * r[i] - 0.4), nrm(n[i] + 0.7 * t[i]), yan[i], 3.4 * o + 0.6, 1.25 * o + 0.3))
dik_taban = np.array([d[0] for d in ds])


def dikenler(P, idx):
    out = np.full(len(P), 1e3)
    for j in np.unique(idx):
        if j >= len(ds):
            continue
        m = idx == j
        b, a, sv, hh, ww = ds[j]
        w = P[m] - b
        tt = w @ a
        rad = np.linalg.norm(w - tt[:, None] * a + (w @ sv)[:, None] * sv * 0.6, axis=1)
        out[m] = np.maximum.reduce([rad - ww * (1 - np.clip(tt / hh, 0, 1)) ** 1.3, -tt, tt - hh]) * 0.7
    return out


agac = cKDTree(C)
dik_agac = cKDTree(dik_taban)


def ejderha(P):
    d = np.full(len(P), 10.0)
    dist, i = agac.query(P, distance_upper_bound=9.0)
    ok = np.isfinite(dist)
    if ok.any():
        Q, j = P[ok], i[ok]
        w = Q - C[j]
        wn, ws, wt = (w * n[j]).sum(1), (w * yan[j]).sum(1), (w * t[j]).sum(1)
        rj = r[j]
        phi = np.arctan2(wn, ws)
        pul = 0.17 * (rj / 4.8 + 0.3) * np.sin(2 * np.pi * yay[j] / 1.8) * np.sin(8 * phi)
        karin = np.where(wn < 0, 0.12 * np.sin(2 * np.pi * yay[j] / 1.3), 0)   # karın halkaları
        d[ok] = np.sqrt(ws ** 2 + (wn / 0.85) ** 2 + wt ** 2) - rj - pul - karin
    dd, di = dik_agac.query(P, distance_upper_bound=5.0)
    ok = np.isfinite(dd)
    if ok.any():
        d[ok] = smin(d[ok], dikenler(P[ok], di[ok]), 0.4)
    for c, rad, fn in parcalar:
        m = np.linalg.norm(P - c, axis=1) < rad
        if m.any():
            d[m] = smin(d[m], fn(P[m]), 0.9)
    # güvenlik: menteşe kulaklarına ve kulpa ejderha girmesin
    ry = np.hypot(P[:, 1] - MY, P[:, 2] - MZ)
    d = np.maximum(d, np.minimum(R_KULAK + 1.2 - ry, np.abs(P[:, 0]) - AX + 3))
    return d


# ---------------- KANATLAR + PUL DOKUSU (kılıf yüzeyinde kabartma) ----------------
KAFES = 2.4
_J = np.random.default_rng(7).uniform(-0.38, 0.38, (48, 48, 48, 3))
_O = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3)


def pul_dokusu(P):
    """Kertenkele derisi gibi karo pullar (3B hücre deseni; her yüzeyde bozulmadan durur)."""
    c = np.floor(P / KAFES).astype(int)
    f1 = np.full(len(P), 1e9)
    f2 = np.full(len(P), 1e9)
    for o in _O:
        cc = c + o
        m = (cc + 0.5 + _J[cc[:, 0] % 48, cc[:, 1] % 48, cc[:, 2] % 48]) * KAFES
        dd = np.linalg.norm(P - m, axis=1)
        f2 = np.where(dd < f1, f1, np.minimum(f2, dd))
        f1 = np.minimum(f1, dd)
    e = np.clip((f2 - f1) / 0.8, 0, 1) ** 0.6
    return 0.45 * e * (1 - 0.3 * (f1 / KAFES) ** 2)


def kanat2d(x, z):
    """Arka yüzde yarasa kanadı kabartması (yükseklik). x>0 tarafı tanımlanır, aynalanır."""
    x = np.abs(x)
    p = np.stack([x, z], -1)
    R_, W_ = np.array([3.0, 14.5]), np.array([12.5, 25.5])
    uclar = [np.array([23.5, 27.0]), np.array([24.0, 19.0]), np.array([20.5, 14.0]), np.array([12.0, 13.0])]

    def seg(a, b):
        ab = b - a
        h = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return np.linalg.norm(p - a - h[..., None] * ab, axis=-1), h

    def ucgen(a, b, c):
        d = np.minimum.reduce([seg(a, b)[0], seg(b, c)[0], seg(c, a)[0]])
        def s_(a, b):
            return (b[0] - a[0]) * (p[..., 1] - a[1]) - (b[1] - a[1]) * (p[..., 0] - a[0])
        s1, s2, s3 = s_(a, b), s_(b, c), s_(c, a)
        ic = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
        return np.where(ic, -d, d)

    zar = ucgen(R_, W_, uclar[-1])
    for a, b in zip(uclar[:-1], uclar[1:]):
        zar = np.minimum(zar, ucgen(W_, a, b))
        mid, L = (a + b) / 2, np.linalg.norm(b - a)
        dis_yon = mid - W_
        dis_yon = dis_yon / np.linalg.norm(dis_yon)
        rc = L * 0.62
        merkez = mid + dis_yon * np.sqrt(max(rc ** 2 - (L / 2) ** 2, 0)) * 0.9
        zar = np.maximum(zar, -(np.linalg.norm(p - merkez, axis=-1) - rc * 0.93))   # oyuk kenar
    damar = 0.12 * np.cos(np.arctan2(p[..., 1] - W_[1], p[..., 0] - W_[0]) * 34)
    hz = (0.6 + damar) * np.clip(-zar / 0.4, 0, 1) + 0.35 * np.clip(1 - np.abs(zar) / 0.35, 0, 1)
    dk, hk = seg(R_, W_)
    hb = 1.6 * np.sqrt(np.clip(1 - (dk / 1.0) ** 2, 0, 1))                          # kol kemiği
    for uc in uclar:
        dp, hp = seg(W_, uc)
        rp = 0.75 - 0.4 * hp
        hb = np.maximum(hb, (1.3 - 0.6 * hp) * np.sqrt(np.clip(1 - (dp / rp) ** 2, 0, 1)))
    hb = np.maximum(hb, 1.9 * np.sqrt(np.clip(1 - (np.linalg.norm(p - W_, axis=-1) / 1.2) ** 2, 0, 1)))  # bilek
    pc = np.linalg.norm(p - (W_ + [1.6, 1.4]), axis=-1)
    hb = np.maximum(hb, np.clip(1.4 - 1.4 * pc / 0.9, 0, None))                    # bilek pençesi
    return np.maximum(hz, hb)


def yukseklik(P, nn):
    h = pul_dokusu(P)
    h = np.where(nn[:, 2] < -0.75, 0.0, h)                         # alt yüz düz kalsın (masada dursun)
    arka = (nn[:, 1] > 0.3) & (P[:, 2] < EK - 3)
    if arka.any():
        h[arka] = np.maximum(h[arka], kanat2d(P[arka, 0], P[arka, 2]))
    return h


# ---------------- MENTEŞE + KULP PARÇALARI ----------------
def eksen(P):
    """Menteşe eksenine göre: |x|, eksene uzaklık, açı (0 = arka, 90 = yukarı, 180 = ön)."""
    dy, dz = P[:, 1] - MY, P[:, 2] - MZ
    return np.abs(P[:, 0]), np.hypot(dy, dz), np.degrees(np.arctan2(dz, dy)) % 360, dy, dz


def dilim(dy, dz, a1, a2):
    """a1→a2 (saat yönü tersine, <180°) açı diliminin işaretli uzaklığı."""
    e1 = np.radians(a1)
    e2 = np.radians(a2)
    d1 = -(np.cos(e1) * dz - np.sin(e1) * dy)
    d2 = np.cos(e2) * dz - np.sin(e2) * dy
    return np.maximum(d1, d2)


def kapak_ek(P):
    ax, ry, _, dy, dz = eksen(P)
    plaka = np.maximum.reduce([ry - R_IC, X0 - ax, ax - (X0 + PLAKA)])
    kol = np.maximum.reduce([ry - 7.0, 3.6 - ry, dilim(dy, dz, 55, 135), (AX - 9) - ax, ax - (X0 + PLAKA)])
    uc = X2 - 0.15
    pim = np.maximum.reduce([ry - (PIM_R - np.maximum(0, ax - (uc - 0.5))), X0 + 0.5 - ax, ax - uc])
    return np.minimum.reduce([plaka, kol, pim])


def govde_ek(P):
    ax, ry, aci, dy, dz = eksen(P)
    kulak = np.maximum.reduce([ry - R_KULAK, X1 - ax, ax - X2])
    dolgu = np.maximum.reduce([ry - R_KULAK, (R_IC + 0.5) - ry, dilim(dy, dz, 180, 270), (AX - 9) - ax, ax - X2,
                               P[:, 2] - (EK - ARALIK)])
    return np.minimum.reduce([kulak, dolgu, kulp(P)])


SLOT = np.radians(ACILMA)
SLOT_YON = np.array([np.sin(SLOT), np.cos(SLOT)])   # (dy, dz): tam açık kapağın "yukarı"sı


def govde_delik(P):
    ax, ry, _, dy, dz = eksen(P)
    yan_ = ax - (X1 - 0.2)
    delik = np.maximum(ry - DELIK_R, -yan_)
    boy = dy * SLOT_YON[0] + dz * SLOT_YON[1]
    en = np.abs(-dy * SLOT_YON[1] + dz * SLOT_YON[0])
    agiz = np.maximum.reduce([en - YUVA / 2, -boy, -yan_])
    return np.minimum(delik, agiz)


HX = AX + DIS + 0.6 + HALKA_DELIK    # delik merkezi (x)


def kulp(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    px, pz = x - HX, z - HALKA_Z
    daire = np.hypot(px, pz) - HALKA_DIS
    damla = np.maximum.reduce([(AX - 3) - x, px, pz - HALKA_DIS, -(pz + HALKA_DIS) + px])  # 45° alt: desteksiz basılır
    d2 = np.minimum(daire, damla)
    dy = np.abs(y - HALKA_Y) - HALKA_KAL / 2
    rr = 1.2
    return np.hypot(np.maximum(d2 + rr, 0), np.maximum(dy + rr, 0)) + np.minimum(np.maximum(d2 + rr, dy + rr), 0) - rr


def kulp_delik(P):
    px, pz = P[:, 0] - HX, P[:, 2] - HALKA_Z
    dy = np.abs(P[:, 1] - HALKA_Y) - HALKA_KAL / 2
    return np.hypot(px, pz) - HALKA_DELIK - np.clip(dy + 0.8, 0, 0.8)   # ağızlarda pah


# ---------------- AÇILMA SÜPÜRMESİ ----------------
ACILAR = np.radians(np.arange(0, ACILMA + 0.1, 4.0))


def dondur(P, a):
    """Kapağı menteşe ekseni etrafında a kadar açar (ön kenar yukarı-arkaya gider)."""
    dy, dz = P[:, 1] - MY, P[:, 2] - MZ
    c, s_ = np.cos(a), np.sin(a)
    return np.c_[P[:, 0], MY + dy * c + dz * s_, MZ - dy * s_ + dz * c]


def kapak_basit(P):
    k = kutu_sdf(P)
    S = np.maximum(k - DIS - 0.5, (EK + ARALIK) - P[:, 2])
    S = np.minimum(S, kapak_ek(P))
    return np.maximum(S, -(k - BOSLUK_KAPAK))


def taban(P):
    """AirPods kutusunun alt yarısı (kapak dönerken buna girmemeli)."""
    return np.maximum(kutu_sdf(P), P[:, 2] - EK)


def supurme(P, fn, yon):
    m = np.full(len(P), 1e3)
    for a in ACILAR[1:]:
        m = np.minimum(m, fn(dondur(P, yon * a)))
    return m


# ---------------- ALAN ----------------
USB = lambda x, y, z: np.maximum.reduce([np.abs(x) - 6.4, np.abs(y) - 3.6, z - 4.0])


def alan(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    k = kutu_sdf(P)
    S = k - DIS
    bant = (S > -1.0) & (S < 2.5)
    if bant.any():
        S[bant] -= yukseklik(P[bant], normal(P[bant]))
    yakin = k < DIS + 18
    if yakin.any():
        S[yakin] = smin(S[yakin], ejderha(P[yakin]), 1.0)
    Sg = np.minimum(np.maximum(S, z - (EK - ARALIK)), govde_ek(P))
    Sk = np.minimum(np.maximum(S, (EK + ARALIK) - z), kapak_ek(P))
    Sg = np.maximum(Sg, -(k - BOSLUK))
    Sk = np.maximum(Sk, -(k - BOSLUK_KAPAK))
    Sg = np.maximum.reduce([Sg, -USB(x, y, z), -np.maximum(np.hypot(x, z - LED_Z) - 1.6, y), -kulp_delik(P), -govde_delik(P)])
    # menteşe bölgesi: kapak açılırken çarpışan yerleri oy
    _, ry, _, dy, _ = eksen(P)
    bolge = (ry < 18) & (dy > -8)
    m = bolge & (Sk < 0.5)
    if m.any():
        Sk[m] = np.maximum(Sk[m], 0.3 - supurme(P[m], taban, 1))      # kapak tabana girmesin
    m = bolge & (Sg < 0.5)
    if m.any():
        Sg[m] = np.maximum(Sg[m], 0.4 - supurme(P[m], kapak_basit, -1))  # kapak gövdeye çarpmasın
    return Sg, Sk


# ---------------- IZGARA + YÜZEY ----------------
if __name__ == "__main__":
    xs = np.arange(-42, 42 + H, H)
    ys = np.arange(-30, 28 + H, H)
    zs = np.arange(-4, 64 + H, H)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    VG = np.empty((len(xs), len(ys), len(zs)), np.float32)
    VK = np.empty_like(VG)
    for kk, zk in enumerate(zs):
        P = np.c_[X.ravel(), Y.ravel(), np.full(X.size, zk)]
        g, kp = alan(P)
        VG[:, :, kk] = g.reshape(X.shape)
        VK[:, :, kk] = kp.reshape(X.shape)
        if kk % 40 == 0:
            print(f"katman {kk}/{len(zs)}", flush=True)
    klasor = os.environ.get("CIKTI", os.path.dirname(os.path.abspath(__file__)))
    if os.environ.get("HACIM"):
        np.savez_compressed(os.environ["HACIM"], VG=VG, VK=VK, xs=xs, ys=ys, zs=zs)
    for ad, V in (("govde", VG), ("kapak", VK)):
        V[[0, -1]] = V[:, [0, -1]] = V[:, :, [0, -1]] = 1.0
        vert, face, _, _ = marching_cubes(V, 0.0, spacing=(H, H, H), allow_degenerate=False)
        vert += [xs[0], ys[0], zs[0]]
        ag = trimesh.Trimesh(vert, face, process=True)
        if ag.volume < 0:
            ag.invert()
        parca = sorted(ag.split(only_watertight=False), key=lambda m: -len(m.faces))
        if len(parca) > 1:
            print(ad, "küçük kopuk parçalar atıldı:", [(len(p.faces), np.round(p.centroid, 1).tolist()) for p in parca[1:6]])
        m = parca[0]
        man = mf.Manifold(mf.Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
        out = m
        if man.status() == mf.Error.NoError and man.num_tri() > 0:
            for tol in (0.02, 0.01, 0.005):          # sadeleştirme ağı bozarsa daha hassas dene
                mm = man.simplify(tol).to_mesh()
                aday = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
                if aday.is_watertight:
                    out = aday
                    break
        else:
            print(ad, "sadeleştirilemedi:", man.status())
        out.export(os.path.join(klasor, f"airpods4-ejderha-{ad}.stl"))
        print(ad, len(out.faces), "üçgen, kapalı:", out.is_watertight, "boyut:", np.round(out.extents, 1))
