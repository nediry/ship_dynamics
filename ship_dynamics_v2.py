import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import dataframe_image as dfi

class Ship_dynamics:
    def __init__(self, length, breadth, draft, wave_amplitude, block_coef,
                 wave_angle, froude_number, fluid_density, gravity):
        self.length = length
        self.breadth = breadth
        self.draft = draft
        self.wave_amplitude = wave_amplitude
        self.block_coef = block_coef
        self.wave_angle = np.deg2rad(wave_angle)
        self.froude_number = froude_number
        self.fluid_density = fluid_density
        self.gravity = gravity
        self.velocity = self.froude_number * np.sqrt(self.gravity * self.length)
        self.a33 = []
        self.b33 = []
        self.F3R = []
        self.F3I = []
        self.F5R = []
        self.F5I = []
    
    def added_mass(self, we):
        a33_x = [.389, .5, .621, .75, 1, 1.25, 1.5, 1.75, 2]
        a33_y = [5, 4.117, 3.494, 3.083, 2.741, 2.705, 2.848, 3.046, 3.239]
        x = we * np.sqrt(self.breadth / (2 * self.gravity))
        a33 = np.interp(x, a33_x, a33_y) * self.breadth \
            * self.draft * self.fluid_density
        return a33
    
    def damping(self, we):
        b33_x = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
        b33_y = [0, 1.567, 2.162, 2.202, 1.919, 1.471, .992, .6, .332]
        x = we * np.sqrt(self.breadth / (2 * self.gravity))
        b33 = np.interp(x, b33_x, b33_y) * self.fluid_density * self.breadth \
            * self.draft / (np.sqrt(self.breadth / (2 * self.gravity)))
        return b33
    
    def calculate_heave_pitch(self, wave_frequency, wave_number):
        we = wave_frequency - self.velocity * wave_number \
           * np.cos(self.wave_angle)
        a33 = self.added_mass(we)
        b33 = self.damping(we)
        
        # heave hareketi Froude-Krylov kuvetti
        F3FK = 2 * self.fluid_density * self.gravity * self.wave_amplitude \
             * self.breadth * np.exp(-wave_number * self.draft) \
             * np.sin(wave_number * self.length / 2) / wave_number
        
        # heave hareketi Difraksiyon kuvetti 
        F3D = 2 * self.wave_amplitude * np.exp(-wave_number * self.draft) \
            * np.sin(wave_number * self.length / 2) * (-a33 * self.gravity \
            + b33 * wave_frequency * 1j / self.wave_amplitude) 
            

        F3 = F3FK + F3D

        # pitch hareketi Froude-Krylov kuvetti 
        F5FK = self.fluid_density * self.gravity * self.wave_amplitude \
             * self.breadth * np.exp(-wave_number * self.draft) \
             * (2 * np.sin(wave_number * self.length / 2) - self.length \
             * wave_number * np.cos(wave_number * self.length / 2)) \
             * 1j / wave_number**2
        
        # pitch hareketi Difraksiyon kuvveti
        F5D = -(a33 * we * 1j + b33) * self.wave_amplitude * wave_frequency \
            * (2 * np.sin(wave_number * self.length / 2) - self.length \
            * wave_number * np.cos(wave_number * self.length / 2)) \
            * np.exp(-wave_number * self.draft) / wave_number**2
        
        F5 = F5FK + F5D
        M = self.length * self.breadth * self.draft \
          * self.block_coef * self.fluid_density
        
        # jirasyon yarıçapı
        kyy = .25 * self.length
        I = M * kyy**2
        
        A33 = a33 * self.length 
        B33 = b33 * self.length
        C33 = self.fluid_density * self.gravity * self.breadth * self.length
        
        A35 = - B33 * self.velocity / we**2
        B35 =  self.velocity * A33
        
        A55 = a33 * self.length**3 / 12 + A33 * self.velocity**2 / we**2
        B55 = b33 * self.length*3 / 12 + self.velocity**2 / we**2 * B33
        C55 = self.fluid_density * self.gravity \
            * self.breadth * self.length**3 / 12
        
        A53 = B33 * self.velocity / we**2
        B53 = - self.velocity * A33
        
        coef = np.array([[-(M + A33)*we**2 + C33, -B33*we, -A35*we**2, -B35*we],      
                         [  B33*we, -(M + A33)*we**2 + C33, B35*we, -A35*we**2],
                         [-A53*we**2, -B53*we, -(I + A55)*we**2 + C55, -B55*we],
                         [  B53*we, -A53*we**2, B55*we, -(I + A55)*we**2 + C55]])
        
        F = np.array([F3.real, F3.imag, F5.real, F5.imag])
        
        resu = np.linalg.solve(coef, F)
        z_real, z_imag = resu[0], resu[1]
        teta_real, teta_imag = resu[2], resu[3]
        
        heave = np.sqrt(z_real**2 + z_imag**2)
        pitch = np.sqrt(teta_real**2 + teta_imag**2)

        return heave, pitch, F, a33, b33
    
    def show_heave_pitch_rao(self, wave_length):
        rao_heave = np.empty_like(wave_length)
        rao_pitch = np.empty_like(wave_length)

        for i in range(len(wave_length)):
            lamda = wave_length[i] * self.length
            wave_frequency = np.sqrt(self.gravity * np.pi * 2 / lamda)
            wave_number = wave_frequency**2 / self.gravity
            heave, pitch, F, a33, b33 = self.calculate_heave_pitch(
                                        wave_frequency, wave_number)
            
            rao_heave[i] = heave / self.wave_amplitude
            rao_pitch[i] = pitch / self.wave_amplitude
            self.F3R.append(F[0])
            self.F3I.append(F[1])
            self.F5R.append(-F[2])
            self.F5I.append(F[3])
            self.a33.append(a33)
            self.b33.append(b33)
            
        plt.figure(figsize=(10, 4), dpi=80)
        plt.grid()
        plt.plot(wave_length, rao_heave)
        plt.title(r"$ROA_z$")
        plt.ylabel(r"$\frac{z}{A}$")
        plt.xlabel(r"$\frac{\lambda}{L}$")
        
        plt.figure(figsize=(10, 4), dpi=80)
        plt.grid()
        plt.plot(wave_length, rao_pitch)
        plt.title(r"$ROA_\theta$")
        plt.ylabel(r"$\frac{\theta}{A}$")
        plt.xlabel(r"$\frac{\lambda}{L}$")
    
    def rms_periyod(self, wave_length):
        lamda = wave_length * self.length
        wave_frequency = np.sqrt(self.gravity * np.pi * 2 / lamda)
        # Geminizin Deniz Durumu 4
        Hs = 1.88
        # Reyleigh dağılımı
        Sw = (8.1e-3 * self.gravity**2 / wave_frequency**5) * np.exp(-0.032 \
           * (self.gravity**2 / Hs**2) / wave_frequency**4)
        m0 = (quad( lambda w: Sw, 0, np.inf ))[0]
        
        # RMS değeri
        RMS = np.sqrt(np.mean( m0**2) )
        
        # Ortalama Merkez Periyodu
        m1 = (quad( lambda wave_frequency: Sw * wave_frequency, 0, np.inf ))[0]
        T1 = 2 * np.pi * (m0 / m1)
        
        # Ortalama Sıfır Geçme Periyodu
        m2 = (quad( lambda wave_frequency: Sw * wave_frequency**2, 0, np.inf ))[0]
        Tz = 2 * np.pi * np.sqrt( m0 / m2 )
        
        # Ortalama Tepeden Tepeye Periyodu
        m4 = (quad( lambda wave_frequency: Sw * wave_frequency**4, 0, np.inf ))[0]
        Tc = 2 * np.pi * np.sqrt( m2 / m4 )

        return RMS, T1, Tz, Tc
    
    def save_table(self, wave_length):
        df = pd.DataFrame([self.a33, self.b33, self.F3R, self.F3I, self.F5R,
                           self.F5I], columns=np.round(wave_length, 2),
                           index=['a33', 'a55', 'F3R', 'F3I', 'F5R', 'F5I'])
        df = df.round(2)
        dfi.export(df, 'tablo2.png')
        
ship = Ship_dynamics(100, 20, 2.5, 1, 1, 135, .1, 1.025, 9.808)

wave_length = np.arange(.25, 6. + .25, step = .25)
ship.show_heave_pitch_rao(wave_length)
ship.save_table(wave_length)

wave_length = 6

RMS, T1, Tz, Tc = ship.rms_periyod(wave_length)