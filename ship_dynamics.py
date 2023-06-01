import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import dataframe_image as dfi

def added_mass(we, B, T, g, rho):
    a33_x = [.389, .5, .621, .75, 1, 1.25, 1.5, 1.75, 2]
    a33_y = [5, 4.117, 3.494, 3.083, 2.741, 2.705, 2.848, 3.046, 3.239]
    x = we * np.sqrt(B / (2 * g))
    a33 = np.interp(x, a33_x, a33_y) * rho * B * T
    return a33

def damping(we, B, T, g, rho):
    b33_x = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
    b33_y = [0, 1.567, 2.162, 2.202, 1.919, 1.471, .992, .6, .332]
    x = we * np.sqrt(B / (2 * g))
    b33 = np.interp(x, b33_x, b33_y) * rho * B * T / (np.sqrt(B / (2 * g)))
    return b33

def calc_heave_pitch(w, k, A, L, B, T, CB, rho, g, V, dalga_aci):
    
    we = w - V * k * np.cos(dalga_aci)
    a33 = added_mass(we, B, T, g, rho)
    b33 = damping(we, B, T, g, rho)
    
    # heave hareketi Froude-Krylov kuvetti
    F3FK = 2 * rho * g * A * B * np.exp(-k * T) * np.sin(k * L / 2) / k
    
    # heave hareketi Difraksiyon kuvetti 
    F3D = -2 * a33 * g * A * np.exp(-k * T) * np.sin(k * L / 2) \
        + 2 * b33 * w * A * np.exp(-k * T) * np.sin(k * L / 2) * 1j / k

    F3 = F3FK + F3D

    # pitch hareketi Froude-Krylov kuvetti 
    F5FK = rho * g * A * B * np.exp(-k * T) * (2 * np.sin(k * L / 2) \
         - L * k * np.cos(k * L / 2)) * 1j / k**2
    
    # pitch hareketi Difraksiyon kuvveti
    F5D = -(a33 * we * 1j + b33) * A * w * (2 * np.sin(k * L / 2) \
        - L * k * np.cos(k * L / 2)) * np.exp(-k * T) / k**2
    
    F5 = F5FK + F5D
    M = L * B * T * CB * rho
    
    # jirasyon yarıçapı
    kyy = .25 * L
    I = M * kyy**2
    
    A33 = a33 * L
    B33 = b33 * L
    C33 = rho * g * B * L
    
    A35 = - B33 * V / we**2
    B35 =  V * A33
    # C35 = C53 = 0
    
    A55 = a33 * L**3 / 12 + A33 * V**2 / we**2
    B55 = b33 * L*3 / 12 + V**2 / we**2 * B33
    C55 = rho * g * B * L**3 / 12
    
    A53 = B33 * V / we**2
    B53 = - V * A33
    
    coef = np.array([[-(M + A33)*we**2 + C33, -B33*we, -A35*we**2, -B35*we],      
                     [  B33*we, -(M + A33)*we**2 + C33, B35*we, -A35*we**2],
                     [-A53*we**2, -B53*we, -(I + A55)*we**2 + C55, -B55*we],
                     [  B53*we, -A53*we**2, B55*we, -(I + A55)*we**2 + C55]])
    
    F = np.array([F3.real, F3.imag, F5.real, F5.imag])
    resu = np.linalg.solve(coef, F)
    z_real, z_imag = resu[0], resu[1]
    teta_real, teta_imag = resu[2], resu[3]
    
    z = np.sqrt(z_real**2 + z_imag**2)
    teta = np.sqrt(teta_real**2 + teta_imag**2)

    return z, teta, a33, b33, F

def rms_periyod(g, w):
    # Geminizin Deniz Durumu 4
    Hs = 1.88
    # Reyleigh dağılımı
    Sw = (8.1e-3 * g**2 / w**5) * np.exp(-0.032 * (g**2 / Hs**2) / w**4)
    m0 = (quad( lambda w: Sw, 0, np.inf ))[0]
    
    # RMS değeri
    RMS = np.sqrt(np.mean( m0**2) )
    
    # Ortalama Merkez Periyodu
    m1 = (quad( lambda w: Sw * w, 0, np.inf ))[0]
    T1 = 2 * np.pi * (m0 / m1)
    
    # Ortalama Sıfır Geçme Periyodu
    m2 = (quad( lambda w: Sw * w**2, 0, np.inf ))[0]
    Tz = 2 * np.pi * np.sqrt( m0 / m2 )
    
    # Ortalama Tepeden Tepeye Periyodu
    m4 = (quad( lambda w: Sw * w**4, 0, np.inf ))[0]
    Tc = 2 * np.pi * np.sqrt( m2 / m4 )

    return RMS, T1, Tz, Tc

L = 100
B = 20
T = 2.5
rho = 1.025
g = 9.808
A = 1
cb = 1
dalga_aci = np.deg2rad(135)
Fr = .1
V = Fr * np.sqrt(g * L)

x = np.arange(.25, 6. + .25, step = .25)
y1, y2 = np.empty_like(x), np.empty_like(x)
a3, b3 = np.empty_like(x), np.empty_like(x)
F3R, F3I = np.empty_like(x), np.empty_like(x)
F5R, F5I = np.empty_like(x), np.empty_like(x)


for i in range(len(x)):
    lamda = x[i] * L
    w = np.sqrt(g * np.pi * 2 / lamda)
    k = w**2 / g
    z, teta, a33, b33, F = calc_heave_pitch(w, k, A, L, B, T, cb,
                                            rho, g, V, dalga_aci)
    
    y1[i] = z / A
    y2[i] = teta / A
    a3[i] = a33
    b3[i] = b33
    F3R[i] = F[0]
    F3I[i] = F[1]
    F5R[i] = F[2]
    F5I[i] = F[3]

plt.figure(figsize=(10, 4), dpi=80)
plt.grid()
plt.plot(x, y1)
plt.title(r"$ROA_z$")
plt.ylabel(r"$\frac{z}{A}$")
plt.xlabel(r"$\frac{\lambda}{L}$")

plt.figure(figsize=(10, 4), dpi=80)
plt.grid()
plt.plot(x, y2)
plt.title(r"$ROA_\theta$")
plt.ylabel(r"$\frac{\theta}{A}$")
plt.xlabel(r"$\frac{\lambda}{L}$")

df = pd.DataFrame([a3, b3, F3R, F3I, -F5R, F5I], columns=np.round(x, 2),
                  index=['a33', 'a55', 'F3R', 'F3I', 'F5R', 'F5I'])
df = df.round(2)
dfi.export(df, 'tablo.png')

RMS, T1, Tz, Tc = rms_periyod(g, w)

del i, k, L, lamda, rho, T, teta, V, w, x, y1, y2, z,
del A, a3, a33, B, b3, b33, g, F, Fr, dalga_aci, cb