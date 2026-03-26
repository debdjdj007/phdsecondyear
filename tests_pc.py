from __future__ import annotations
from typing import Optional
import math
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction

class AugmentedRosenbrock(SyntheticTestFunction):
    r"""Augmented Rosenbrock for multi-fidelity optimization.

    d-dimensional function (evaluate on [-5, 10]^(d-2) * [0, 1]^2),
    last two dimensions are fidelity parameters.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 3,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        if dim < 3:
            raise ValueError("AugmentedRosenbrock needs dim >= 3")
        self.dim = dim
        self.continuous_inds = list(range(dim))
        # (d-2) design vars in [-5,10], then 2 fidelity dims in [0,1]
        self._bounds = [(-5.0, 10.0) for _ in range(self.dim - 2)] + [(0.0, 1.0), (0.0, 1.0)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    # BoTorch 0.10.0 requires this name:
    def evaluate_true(self, X: Tensor) -> Tensor:
        # design dims are everything except the last 2 fidelities
        # this matches the latest BoTorch implementation
        X_curr = X[..., :-3]
        X_next = X[..., 1:-2]
        t1 = 100 * (X_next - X_curr.pow(2) + 0.1 * (1 - X[..., -2:-1])).pow(2)
        t2 = (X_curr - 1 + 0.1 * (1 - X[..., -1:]).pow(2)).pow(2)
        return (t1 + t2).sum(dim=-1)



import os
import numpy as np
import torch
from olympus.emulators import Emulator
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

import sys
import types
from pathlib import Path
import numpy as np
import matplotlib.cm as cm

def patch_olympus_windows():
    # Avoid mixed old user-site install
    bad_site = r"C:\Users\ghosh\AppData\Roaming\Python\Python39\site-packages"
    sys.path = [p for p in sys.path if bad_site.lower() not in p.lower()]

    # np.delete ragged fix (Olympus dataset split compatibility)
    if not getattr(np, "_olympus_delete_patch", False):
        _orig_delete = np.delete

        def _delete_ragged_ok(arr, obj, axis=None):
            try:
                return _orig_delete(arr, obj, axis=axis)
            except ValueError:
                if axis == 0 and isinstance(arr, (list, tuple)):
                    return _orig_delete(np.array(arr, dtype=object), obj, axis=axis)
                raise

        np.delete = _delete_ragged_ok
        np._olympus_delete_patch = True

    # Avoid duplicate cmap crash on repeated imports
    if not getattr(cm, "_olympus_safe_register_patch", False):
        _orig_register = cm.register_cmap

        def _safe_register_cmap(*args, **kwargs):
            try:
                return _orig_register(*args, **kwargs)
            except ValueError as exc:
                if "already registered" in str(exc):
                    return None
                raise

        cm.register_cmap = _safe_register_cmap
        cm._olympus_safe_register_patch = True

    # silence_tensorflow shim if missing
    if "silence_tensorflow" not in sys.modules:
        st = types.ModuleType("silence_tensorflow")
        st.silence_tensorflow = lambda *a, **k: None
        sys.modules["silence_tensorflow"] = st

    import olympus.datasets as ods
    import olympus.datasets.dataset as ds
    import olympus.models as om
    import olympus.models.model as model_mod
    import olympus.emulators.emulator as emu_mod

    # Dataset patch (Windows-safe discovery)
    ds_root = Path(ds.__file__).resolve().parent

    def _list_datasets_windows():
        return sorted(
            p.name[len("dataset_"):]
            for p in ds_root.glob("dataset_*")
            if p.is_dir() and p.name.startswith("dataset_")
        )

    def _validate_dataset_args_windows(kind, data, columns, target_names):
        if kind is None:
            return
        avail = _list_datasets_windows()
        if kind not in avail:
            ds.Logger.log(
                f"Could not find dataset `{kind}`. Please choose from: {', '.join(avail)}.",
                "FATAL",
            )

    ods.datasets_list = _list_datasets_windows()
    ods.list_datasets = _list_datasets_windows
    ds._validate_dataset_args = _validate_dataset_args_windows
    emu_mod._validate_dataset_args = _validate_dataset_args_windows

    # Model patch (Windows-safe discovery)
    models_home = Path(getattr(om, "__home__", Path(model_mod.__file__).resolve().parent))

    def _snake_to_camel(name):
        return "".join(part.capitalize() for part in name.split("_"))

    def _get_models_list_windows():
        return sorted(
            _snake_to_camel(p.name[len("model_"):])
            for p in models_home.glob("model_*")
            if p.is_dir() and p.name.startswith("model_")
        )

    def _validate_model_kind_windows(kind):
        if type(kind) == str:
            avail = _get_models_list_windows()
            if kind not in avail:
                model_mod.Logger.log(
                    f'Model "{kind}" not available. Choose from: {", ".join(avail)}',
                    "FATAL",
                )
        elif issubclass(kind, model_mod.AbstractModel):
            return
        else:
            model_mod.Logger.log("Invalid model kind", "FATAL")

    om.model_names = _get_models_list_windows()
    om.get_models_list = _get_models_list_windows
    model_mod.get_models_list = _get_models_list_windows
    model_mod._validate_model_kind = _validate_model_kind_windows
    emu_mod._validate_model_kind = _validate_model_kind_windows

patch_olympus_windows()

# pick model safely
try:
    import tensorflow_probability  # noqa: F401
    OLYMPUS_MODEL = "BayesNeuralNet"
except ModuleNotFoundError:
    OLYMPUS_MODEL = "NeuralNet"


class SuzukiHFEmulator:
    """
    High-fidelity Suzuki from Olympus emulator.
    Uses input format [x1,x2,x3,x4,s], but s is fixed to 1 via bounds.
    """
    def __init__(self, num_samples: int = 16, clip=(0.0, 100.0)):
        os.environ.setdefault("OLYMPUS_SCRATCH", "/tmp/olympus_scratch")
        os.makedirs(os.environ["OLYMPUS_SCRATCH"], exist_ok=True)

        self.emu = Emulator(dataset="suzuki", model=OLYMPUS_MODEL)
        self.num_samples = num_samples
        self.clip = clip

        self.x_lb = np.array([p.low for p in self.emu.param_space], dtype=float)
        self.x_ub = np.array([p.high for p in self.emu.param_space], dtype=float)

        # Keep MF interface (last dim is fidelity), but fix s=1.0 for pure HF benchmark
        self.dim = 5
        lb = np.concatenate([self.x_lb, [1.0]])
        ub = np.concatenate([self.x_ub, [1.0]])
        self.bounds = torch.tensor(np.vstack([lb, ub]), **tkwargs)

    def _hf_scalar(self, x):
        y = float(self.emu.run(np.asarray(x, dtype=float), num_samples=self.num_samples)[0, 0])
        return float(np.clip(y, self.clip[0], self.clip[1]))

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.detach().cpu().numpy()
        ys = [self._hf_scalar(row[:4]) for row in X_np]
        return torch.tensor(ys, **tkwargs)


class SuzukiMFEmulator(SuzukiHFEmulator):
    """
    Continuous-fidelity Suzuki using your mf_observe logic.
    Input: [x1,x2,x3,x4,s], s in [0,1].
    """
    def __init__(self, num_samples: int = 16, clip=(0.0, 100.0), stochastic: bool = False):
        super().__init__(num_samples=num_samples, clip=clip)
        self.stochastic = stochastic

        # For MF, fidelity ranges in [0,1]
        lb = np.concatenate([self.x_lb, [0.0]])
        ub = np.concatenate([self.x_ub, [1.0]])
        self.bounds = torch.tensor(np.vstack([lb, ub]), **tkwargs)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        X_np = X.detach().cpu().numpy()
        ys = []

        for row in X_np:
            x = row[:4]
            s = float(np.clip(row[4], 0.0, 1.0))

            y_hi_true = self._hf_scalar(x)

            x01 = (x - self.x_lb) / (self.x_ub - self.x_lb + 1e-12)
            phi = np.array([
                np.sin(2 * np.pi * x01[1]),
                np.cos(2 * np.pi * x01[2]),
                np.sin(2 * np.pi * x01[3]),
                np.cos(2 * np.pi * x01[0]),
            ])
            x_tilde = np.clip(x01 + 0.07 * (1.0 - s) ** 1.2 * phi, 0.0, 1.0)

            x_warp = self.x_lb + x_tilde * (self.x_ub - self.x_lb)
            y_low_base = self._hf_scalar(x_warp)

            y_ref = s * y_hi_true + (1.0 - s) * y_low_base

            delta = (1.0 - s) ** 1.1 * (
                10.0 * np.sin(2 * np.pi * x_tilde[0] * x_tilde[2])
                - 7.0 * (x_tilde[1] - 0.45) ** 2
                + 5.0 * np.cos(3 * np.pi * x_tilde[3])
            )

            eps = 0.0
            if self.stochastic:
                sigma = 0.25 + 3.5 * (1.0 - s) ** 1.3 + 1.2 * (1.0 - s) * (
                    abs(x_tilde[0] - 0.5) + abs(x_tilde[2] - 0.5)
                )
                eps = np.random.normal(0.0, sigma)
                if np.random.rand() < (0.01 + 0.08 * (1.0 - s) ** 1.5):
                    eps += np.random.normal(0.0, 6.0 * (1.0 - s))

            y_mf = np.clip(y_ref + delta + eps, self.clip[0], self.clip[1])
            ys.append(float(y_mf))

        return torch.tensor(ys, **tkwargs)



