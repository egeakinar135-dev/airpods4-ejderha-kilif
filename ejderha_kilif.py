"""AirPods 4 kılıfı: etrafına sarılmış ejderha. İki parça (gövde + kapak) STL üretir.

Kullanım:  python ejderha_kilif.py
Ölçüler mm. Kılıf oturmazsa önce aşağıdaki ÖLÇÜLER ve BOŞLUK değerlerini düzelt.
"""
import os
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh
import manifold3d as mf

# ---------------- ÖLÇÜLER (AirPods 4 şarj kutusu, Apple: 46,2 × 50,1 × 21,2 mm) ----------------
GEN, YUK, DER = 50.1, 46.2, 21.2   # genişlik (x), yükseklik (z), derinlik (y)
R_ON = 14.0      # önden bakınca köşe yuvarlaklığı
R_KENAR = 7.0    # yan kenarların yuvarlaklığı (derinlik yönünde)
EK_YERI = 32.0   # kapak çizgisinin alttan yüksekliği (kendi kutunda ölç!)
LED_Z = 26.0     # ön ışık deliğinin alttan yüksekliği

# ---------------- BASKI ----------------
BOSLUK = 0.20    # kutu ile kılıf arası (PLA: 0.20, TPU: 0.0-0.05)
KALINLIK = 1.6   # kılıf duvarı
ARALIK_ON = 0.35   # gövde-kapak arası boşluk (yarısı)
ARALIK_ARKA = 1.5  # menteşe tarafındaki boşluk (yarısı)
H = 0.2          # çözünürlük (küçük = daha detaylı, daha yavaş)

AX, AY, AZ = GEN / 2, DER / 2, YUK / 2
DIS = BOSLUK + KALINLIK


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
    for _ in range(3):
        n = normal(q)
        q = q - n * (kutu_sdf(q) - DIS - kalk)[..., None]
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


# ---------------- EJDERHANIN OMURGASI ----------------
N = 3000
s = np.linspace(0, 1, N)
TUR = 1.4 * 2 * np.pi
aci = -0.2 + s * TUR                       # 0 = ön orta, +π/2 = sağ, π = arka
z = np.interp(s, [0, 0.07, 0.15, 0.22, 1.0], [39.5, 35.5, 25.5, 24, 6])  # kapak çizgisini önde geçer
zarf = np.clip((s - 0.22) / 0.1, 0, 1)
z = z + zarf * 2.2 * np.sin(2 * np.pi * 2.5 * (s - 0.22) / 0.78)
r = np.interp(s, [0, 0.06, 0.2, 0.4, 1.0], [2.8, 3.0, 3.4, 3.4, 0.9])

yon = np.stack([np.sin(aci), -np.cos(aci), np.zeros(N)], 1)
lo, hi = np.zeros(N), np.full(N, 45.0)
for _ in range(40):                        # kılıf yüzeyini ışınla bul
    m = (lo + hi) / 2
    ic = kutu_sdf(yon * m[:, None] + np.c_[0 * z, 0 * z, z]) < DIS
    lo, hi = np.where(ic, m, lo), np.where(ic, hi, m)
yuz = yon * lo[:, None] + np.c_[0 * z, 0 * z, z]
n = normal(yuz)
C = yuz + n * (0.35 * r)[:, None]          # gövde yarı gömülü
t = nrm(np.gradient(C, axis=0))
yan = nrm(np.cross(n, t))
n = np.cross(t, yan)
yay = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]

# ---------------- PARÇALAR ----------------
parcalar = []  # (merkez, yarıçap, sdf fonksiyonu) — sadece yakın noktalarda hesaplanır

# kafa
t0, n0, c0 = t[0], n[0], C[0]
f = nrm(-t0 + 0.35 * n0)
u = nrm(n0 - (n0 @ f) * f)
sh = np.cross(f, u)
KB = 1.35       # kafa büyüklüğü
K = c0 + f * 1.5 * KB


def kafa(P):
    return KB * _kafa(K + (P - K) / KB)


def _kafa(P):
    d = elipsoid(P, K, np.stack([f, sh, u]), np.array([4.0, 3.4, 2.6]))                # kafatası
    d = smin(d, kapsul(P, K + f * 1.5 + u * 0.3, K + f * 7.5 + u * 0.1, 2.3, 1.25), 1.0)  # kama burun
    d = smin(d, kapsul(P, K + f * 0.5 - u * 0.9, K + f * 6.8 - u * 1.0, 1.9, 1.05), 0.5)  # alt çene
    d = smin(d, kapsul(P, K + f * 6.2 + u * 1.15, K + f * 0.8 + u * 2.3, 0.45, 0.55), 0.4)  # burun sırtı
    for k in (1, -1):
        d = smin(d, kapsul(P, K + f * 2.6 + u * 1.7 + sh * k * 1.7, K - f * 0.6 + u * 2.0 + sh * k * 2.7, 0.55, 0.75), 0.4)  # kaş
        d = np.minimum(d, np.linalg.norm(P - (K + f * 1.5 + u * 1.0 + sh * k * 2.45), axis=1) - 0.75)  # göz
        m1 = K - f * 1.6 + u * 1.6 + sh * k * 1.6
        m2 = K - f * 6.5 + u * 2.5 + sh * k * 2.5
        m3 = K - f * 10.0 + u * 4.0 + sh * k * 2.9
        d = smin(d, np.minimum(kapsul(P, m1, m2, 1.25, 0.75), kapsul(P, m2, m3, 0.75, 0.2)), 0.5)  # boynuz
        d = smin(d, kapsul(P, K - f * 0.8 + sh * k * 3.0, K - f * 3.6 + sh * k * 4.3 - u * 0.3, 0.6, 0.15), 0.4)  # yanak dikeni
        d = smin(d, kapsul(P, K + f * 4.5 - u * 1.2 + sh * k * 1.35, K + f * 4.3 - u * 2.2 + sh * k * 1.35, 0.4, 0.1), 0.2)  # diş
        d = np.maximum(d, -(np.linalg.norm(P - (K + f * 7.1 + u * 0.8 + sh * k * 0.7), axis=1) - 0.45))  # burun deliği
    return d


parcalar.append((K, 16.0 * KB, kafa))


# bacaklar
def bacak_ekle(i, ileri):
    for k in (1, -1):
        c, ni, ti, si = C[i], n[i], t[i], yan[i] * k
        omuz = c + si * r[i] * 0.5
        diz = c + si * (r[i] + 2.2) + ni * 1.2 + ti * ileri * 0.8
        ayak = yuzeye(c + si * (r[i] + 4.2) + ti * ileri * 1.8, 0.7)
        parmak = [yuzeye(ayak + nrm(si * 1.0 + ti * a) * 2.4, 0.35) for a in (-0.9, 0.0, 0.9)]

        def bacak(P, omuz=omuz, diz=diz, ayak=ayak, parmak=parmak, rr=r[i]):
            d = smin(kapsul(P, omuz, diz, rr * 0.45, 1.05), kapsul(P, diz, ayak, 1.05, 0.9), 0.6)
            for p in parmak:
                d = smin(d, kapsul(P, ayak, p, 0.6, 0.22), 0.3)
            return d

        parcalar.append((ayak, 10.0, bacak))


bacak_ekle(int(0.075 * N), -1)
bacak_ekle(int(0.50 * N), 1)

# kuyruk ucu (mızrak)
ce, te, ne_, se = C[-1] + t[-1] * 1.8, t[-1], n[-1], yan[-1]


def mizrak(P):
    w = P - ce
    a, b = 3.6, 2.6
    q = np.abs(w @ te) * b + np.abs(w @ se) * a - a * b
    return np.maximum(q / np.hypot(a, b), np.abs(w @ ne_) - 0.9)


parcalar.append((ce, 7.0, mizrak))

# sırt dikenleri
ds = []
for L in np.arange(yay[-1] * 0.03, yay[-1] * 0.97, 2.6):
    i = np.searchsorted(yay, L)
    o = r[i] / 3.4
    ds.append((C[i] + n[i] * (0.85 * r[i] - 0.4), nrm(n[i] + 0.55 * t[i]), yan[i], 2.4 * o + 0.5, 0.95 * o + 0.3))
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
        out[m] = np.maximum.reduce([rad - ww * (1 - np.clip(tt / hh, 0, 1)), -tt, tt - hh]) * 0.7
    return out


agac = cKDTree(C)
dik_agac = cKDTree(dik_taban)


def ejderha(P):
    d = np.full(len(P), 10.0)
    dist, i = agac.query(P, distance_upper_bound=8.0)
    ok = np.isfinite(dist)
    if ok.any():
        Q, j = P[ok], i[ok]
        w = Q - C[j]
        wn, ws, wt = (w * n[j]).sum(1), (w * yan[j]).sum(1), (w * t[j]).sum(1)
        rj = r[j]
        phi = np.arctan2(wn, ws)
        pul = 0.13 * (rj / 3.4 + 0.3) * np.sin(2 * np.pi * yay[j] / 1.5) * np.sin(7 * phi)
        d[ok] = np.sqrt(ws ** 2 + (wn / 0.85) ** 2 + wt ** 2) - rj - pul
    dd, di = dik_agac.query(P, distance_upper_bound=4.0)
    ok = np.isfinite(dd)
    if ok.any():
        d[ok] = smin(d[ok], dikenler(P[ok], di[ok]), 0.4)
    for c, rad, fn in parcalar:
        m = np.linalg.norm(P - c, axis=1) < rad
        if m.any():
            d[m] = smin(d[m], fn(P[m]), 0.8)
    return d


def kilif(P):
    k = kutu_sdf(P)
    S = k - DIS
    yakin = k < DIS + 14
    if yakin.any():
        S[yakin] = smin(S[yakin], ejderha(P[yakin]), 1.0)
    S = np.maximum(S, -(k - BOSLUK))                                  # içi boş
    x, y, zz = P[:, 0], P[:, 1], P[:, 2]
    usb = np.maximum.reduce([np.abs(x) - 6.4, np.abs(y) - 3.6, zz - 4.0])
    S = np.maximum(S, -usb)                                           # USB-C
    led = np.maximum(np.hypot(x, zz - LED_Z) - 1.6, y)
    S = np.maximum(S, -led)                                           # ön ışık
    S = np.maximum(S, -(np.abs(zz - EK_YERI) - ARALIK_ON))            # kapak boşluğu
    S = np.maximum(S, -np.maximum(np.abs(zz - EK_YERI) - ARALIK_ARKA, 1.0 - y))  # menteşe
    return S


# ---------------- IZGARA + YÜZEY ----------------
xs = np.arange(-38, 38 + H, H)
ys = np.arange(-27, 24 + H, H)
zs = np.arange(-4, 54 + H, H)
X, Y = np.meshgrid(xs, ys, indexing="ij")
V = np.empty((len(xs), len(ys), len(zs)), np.float32)
for k, zk in enumerate(zs if not os.environ.get("HACIM") else []):
    P = np.c_[X.ravel(), Y.ravel(), np.full(X.size, zk)]
    V[:, :, k] = kilif(P).reshape(X.shape)
    if k % 50 == 0:
        print(f"katman {k}/{len(zs)}", flush=True)
if os.environ.get("HACIM"):
    V = np.load(os.environ["HACIM"])
elif os.environ.get("HACIM_KAYDET"):
    np.save(os.environ["HACIM_KAYDET"], V)
V[[0, -1]] = V[:, [0, -1]] = V[:, :, [0, -1]] = 1.0

vert, face, _, _ = marching_cubes(V, 0.0, spacing=(H, H, H), allow_degenerate=False)
vert += [xs[0], ys[0], zs[0]]
ag = trimesh.Trimesh(vert, face, process=True)
if ag.volume < 0:
    ag.invert()
govdeler = sorted(ag.split(only_watertight=False), key=lambda m: -len(m.faces))[:2]

klasor = os.path.dirname(os.path.abspath(__file__))
for m in govdeler:
    ad = "kapak" if m.centroid[2] > EK_YERI else "govde"
    man = mf.Manifold(mf.Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
    if man.status() == mf.Error.NoError and man.num_tri() > 0:
        mm = man.simplify(0.02).to_mesh()
        out = trimesh.Trimesh(mm.vert_properties[:, :3], mm.tri_verts)
    else:
        print(ad, "sadeleştirilemedi:", man.status())
        out = m
    out.export(os.path.join(klasor, f"airpods4-ejderha-{ad}.stl"))
    print(ad, len(out.faces), "üçgen, kapalı:", out.is_watertight, "boyut:", np.round(out.extents, 1))
