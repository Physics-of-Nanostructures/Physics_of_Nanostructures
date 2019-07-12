import numpy as np
from dataclasses import dataclass

π = np.pi

@dataclass
class StonerWohlfarthModel:
    Ms: float = 0
    Ku: float = 0
    Nx: float = 0
    Keb: float = 0
    θeb: float = 0
    ϕeb: float = 0

    μ0 = 4 * π * 1e-7
    θlist = []
    ϕlist = []
    dθ = 0.01 * π
    dϕ = 0.01 * π

    def calculate_energy_2D(self, θ, θ_field=0, B=0):
        H = B / self.μ0

        E_field = -self.μ0 * self.Ms * H * np.cos(θ_field - θ)
        E_demag = .5 * self.μ0 * self.Ms**2 * np.cos(θ)**2
        E_K_uni = self.Ku * np.sin(θ)**2

        E = E_field + E_demag + E_K_uni

        return E

    def calculate_energy_3D(self, θ, ϕ, θ_field=0, ϕ_field=0, B=0):
        E_field = -B * self.Ms * (
            np.cos(θ) * np.cos(θ_field) +
            np.cos(ϕ - ϕ_field) * np.sin(θ) * np.sin(θ_field)
        )

        E_demag_x = .5 * self.μ0 * self.Ms**2 * np.sin(θ)**2 * np.cos(ϕ)**2
        E_demag_z = .5 * self.μ0 * self.Ms**2 * np.cos(θ)**2
        E_K = self.Ku * np.sin(θ)**2
        
        E_eb = -self.Keb * (
            np.cos(θ) * np.cos(self.θeb) +
            np.cos(ϕ - self.ϕeb) * np.sin(θ) * np.sin(self.θeb)
        )

        E = E_field + E_demag_z + self.Nx * E_demag_x + E_K + E_eb

        return E

    def energy_3D_minimizer(self, angles, *args, **kwargs):
        self.θlist.append(angles[0])
        self.ϕlist.append(angles[1])
        # print(angles)
        return self.calculate_energy_3D(angles[0], angles[1], *args, **kwargs)


def radians_format_func(val, pos):
    red_val = val / π
    if red_val == 0:
        return "0"
    elif red_val % 1 == 0:
        return f"{red_val:.0f}$\pi$"
    elif np.round(2 * red_val, 10) % 1 == 0:
        return r"$\frac{" + f"{2*red_val:.0f}" + r"}{2}\,\pi$"
    elif np.round(3 * red_val, 10) % 1 == 0:
        return r"$\frac{" + f"{3*red_val:.0f}" + r"}{3}\,\pi$"
    elif np.round(6 * red_val, 10) % 1 == 0:
        return r"$\frac{" + f"{6*red_val:.0f}" + r"}{6}\,\pi$"
    else:
        return f"{red_val:.2f}$\pi$"
