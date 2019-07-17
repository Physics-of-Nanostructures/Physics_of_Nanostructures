from tmm.tmm_core import (coh_tmm, position_resolved,
                          find_in_structure_with_inf)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class TMM_stack:
    stack: list
    ambient: tuple = ("Air", 1)
    substrate: list = field(
        default_factory=lambda: [
            ("SiO", 100, 1.4608 + 0.0013066j),
            ("Si", 5e5, 3.6750 + 0.0054113j)
        ]
    )

    back_to_front: bool = True
    back_illumniation: bool = True
    θ0: float = 0
    λ0: float = 800
    polarisation: str = "p"
    plot_margin_l: float = 2
    plot_margin_r: float = 2

    _prepared_layers = False
    _colormap = {}
    _sourcemap = ["gray", "red", "blue", "orange", "green",
                  "magenta", "cyan", "yellow", "purple", ]

    def __post_init__(self):
        if self.polarisation not in ["p", "s"]:
            raise ValueError("polarisation can only be 'p' or 's'")

    def prepare_layers(self):
        if self.back_to_front:
            self.stack = self.stack[::-1]

        if not isinstance(self.ambient, list):
            self.ambient = [self.ambient]

        if not isinstance(self.substrate, list):
            self.substrate = [self.substrate]

        full_stack = [
            *self.ambient,
            *self.stack,
            *self.substrate,
            *self.ambient,
        ]

        self.layers_materials = []
        self.layers_thickness = []
        self.layers_refractive = []

        for layer in full_stack:
            material = layer[0]
            index = layer[-1]

            if len(layer) == 3:
                thickness = layer[1]
            else:
                thickness = np.inf

            self.layers_materials.append(material)
            self.layers_thickness.append(thickness)
            self.layers_refractive.append(index)

        self.total_thickness = np.sum([l[1] for l in self.stack])
        self.materials = list(set(self.layers_materials))

        self._prepared_layers = True

    def calculate_tmm(self):
        if not self._prepared_layers:
            self.prepare_layers()

        coh_tmm_data = coh_tmm(
            self.polarisation,
            self.layers_refractive,
            self.layers_thickness,
            self.θ0, self.λ0
        )

        self.T_step = 0.01
        self.T = np.arange(-self.plot_margin_l,
                           self.total_thickness + self.plot_margin_r + self.T_step,
                           self.T_step)  # position in structure

        self.absorption = np.zeros(len(self.T))
        for idx, t in enumerate(self.T):
            self.absorption[idx] = (
                position_resolved(
                    *find_in_structure_with_inf(self.layers_thickness, t),
                    coh_tmm_data
                )['absor']
            )

    def plot(self, *, ax=None, show_layers=True, label=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
            self.ax.set_ylabel('Absorption (%)')
            self.ax.set_xlabel('Depth (nm)')
        else:
            self.ax = ax

        self.ax.plot(self.T, self.absorption * 100, label=label)

        if show_layers:
            t0 = 0
            for layer in self.stack:
                t = layer[1]

                color = self.mat2col(layer[0])
                self.ax.axvspan(t0, t0 + t, alpha=0.3,
                                color=color, lw=0, zorder=-1)

                self.ax.text(t0 + t / 2, self.ax.get_ylim()[0] * 0.9,
                             f"{layer[0]} ({layer[1]})",
                             ha='center', va='bottom')

                t0 += t

    def mat2col(self, material):
        if material not in self._colormap:
            self._colormap[material] = self._sourcemap.pop(0)

        return self._colormap[material]
