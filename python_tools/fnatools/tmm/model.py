from tmm.tmm_core import (coh_tmm, position_resolved,
                          find_in_structure_with_inf)

import numpy
import pandas
from scipy import integrate
from matplotlib import pyplot
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
    back_illumination: bool = True
    θ0: float = 0
    λ0: float = 800
    polarisation: str = "p"
    plot_margin_l: float = 2
    plot_margin_r: float = 2

    _prepared_layers = False
    _prepared_d_resolved = False
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
        ]

        if len(full_stack[-1]) == 3:
            full_stack.append(*self.ambient)

        self.layers_material = []
        self.layers_thickness = []
        self.layers_refractive = []

        for idx, layer in enumerate(full_stack):
            material = layer[0]
            index = layer[-1]

            if len(layer) == 3:
                thickness = layer[1]
            else:
                thickness = numpy.inf

            self.layers_material.append(material)
            self.layers_thickness.append(thickness)
            self.layers_refractive.append(index)

        self.total_thickness = numpy.sum([l[1] for l in self.stack])
        self.materials = list(set(self.layers_material))

        self._prepared_layers = True
        self._prepared_d_resolved = False

    def prepare_d_resolved(self):
        self.d_step = 0.01
        self.d_res = pandas.DataFrame()
        self.d_res["x"] = numpy.arange(
            -self.plot_margin_l,
            self.total_thickness + self.plot_margin_r + self.d_step,
            self.d_step
        )

        bins = [
            -self.plot_margin_l - 1, 0,
            *numpy.cumsum(self.layers_thickness[1:])
        ]
        labels = [l + f"_{i:d}" for i, l in enumerate(self.layers_material)]
        self.d_res["material"] = pandas.cut(
            self.d_res["x"], bins=bins,
            labels=labels
        )
        self._prepared_d_resolved = True

    def calculate_tmm(self):
        if not self._prepared_layers:
            self.prepare_layers()

        if not self._prepared_d_resolved:
            self.prepare_d_resolved()

        coh_tmm_data = coh_tmm(
            self.polarisation,
            self.layers_refractive,
            self.layers_thickness,
            self.θ0, self.λ0
        )

        self.transmission = coh_tmm_data["T"]
        self.reflection = coh_tmm_data["R"]

        self.d_res["absorption"] = self.d_res["x"].apply(
            lambda x: position_resolved(
                *find_in_structure_with_inf(self.layers_thickness, x),
                coh_tmm_data
            )['absor']
        )

        self.d_res["absorption_cumulative"] = integrate.cumtrapz(
            self.d_res["absorption"], self.d_res["x"], initial=0)

    def plot(self, *, ax=None, show_layers=True, label=None,
             plot_cumulative=True):
        if ax is None:
            self.fig, self.ax = pyplot.subplots(1, 1, figsize=(11.69, 8.27))
            self.ax.set_ylabel('Absorption (%)')
            self.ax.set_xlabel('Depth (nm)')
        else:
            self.ax = ax

        self.ax.plot(self.d_res.x, self.d_res.absorption * 100, label=label)

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

        if plot_cumulative:
            self.ax2 = self.ax.twinx()

            self.ax2.plot(
                self.d_res.x, self.d_res.absorption_cumulative * 100,
                color='r', label=label
            )

            if "absorption_specific" in self.d_res:
                self.ax2.plot(
                    self.d_res.x, self.d_res.absorption_specific * 100,
                    color='r', label=label + " Specific"
                )

    def mat2col(self, material):
        if material not in self._colormap:
            self._colormap[material] = self._sourcemap.pop(0)

        return self._colormap[material]

    def calculate_material_absorption(self, material):
        mask = self.d_res.material.str.contains(material)
        self.d_res["absorption_specific"] = integrate.cumtrapz(
            self.d_res.absorption * mask, self.d_res.x, initial=0)

        absorption_specific = integrate.trapz(
            self.d_res.absorption * mask, self.d_res.x)

        return absorption_specific
