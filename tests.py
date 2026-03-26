from __future__ import annotations
from typing import Optional
import inspect
import math
import torch
from torch import Tensor
from botorch.test_functions.multi_objective import Penicillin as BoTorchPenicillin
from botorch.test_functions.synthetic import Ackley, SyntheticTestFunction


def _init_synthetic_test_function(
    problem: SyntheticTestFunction,
    noise_std: Optional[float] = None,
    negate: bool = False,
    bounds: list[tuple[float, float]] | None = None,
    dtype: torch.dtype = torch.double,
) -> None:
    init_params = inspect.signature(SyntheticTestFunction.__init__).parameters
    kwargs = {"noise_std": noise_std, "negate": negate}
    if bounds is not None:
        problem._bounds = bounds
        if "bounds" in init_params:
            kwargs["bounds"] = bounds
    if "dtype" in init_params:
        kwargs["dtype"] = dtype
    SyntheticTestFunction.__init__(problem, **kwargs)

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


class AugmentedForrester(SyntheticTestFunction):
    r"""Continuous-fidelity Forrester benchmark.

    The input is `(x, s)` with `x in [0, 1]` and fidelity `s in [0, 1]`.
    At `s = 1`, this recovers the standard high-fidelity Forrester function
    `f_hi(x) = (6x - 2)^2 sin(12x - 4)`.

    At `s = 0`, this matches the common low-fidelity surrogate
    `0.5 f_hi(x) + 10(x - 0.5) - 5`. Intermediate values of `s` interpolate
    smoothly between the two, following the same "augmented benchmark" pattern
    used by BoTorch's built-in multi-fidelity test functions.
    """

    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = -6.020740055
    _optimizers = [(0.75725, 1.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        _init_synthetic_test_function(
            self,
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

    @staticmethod
    def _forrester_high(x: Tensor) -> Tensor:
        return (6.0 * x - 2.0).pow(2) * torch.sin(12.0 * x - 4.0)

    def _eval(self, X: Tensor) -> Tensor:
        x = X[..., 0]
        s = X[..., 1].clamp(0.0, 1.0)
        f_hi = self._forrester_high(x)
        return (0.5 + 0.5 * s) * f_hi + (1.0 - s) * (10.0 * (x - 0.5) - 5.0)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)


class AugmentedAckley(SyntheticTestFunction):
    r"""Continuous-fidelity Ackley benchmark.

    The first `dim` coordinates are the Ackley design variables and the final
    coordinate is a continuous fidelity parameter `s in [0, 1]`. At `s = 1`,
    this matches BoTorch's Ackley function exactly.

    Lower fidelities reduce the radial decay coefficient, producing a smoother
    landscape while keeping the same global basin structure.
    """

    _optimal_value = 0.0
    _check_grad_at_opt = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        if dim < 1:
            raise ValueError("AugmentedAckley needs dim >= 1")
        base = Ackley(dim=dim)
        self.design_dim = dim
        self.dim = dim + 1
        self.continuous_inds = list(range(self.dim))
        bounds = list(base._bounds) + [(0.0, 1.0)]
        self._bounds = bounds
        self._optimizers = [tuple([0.0 for _ in range(dim)] + [1.0])]
        self.a = base.a
        self.b = base.b
        self.c = base.c
        _init_synthetic_test_function(
            self,
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
            dtype=dtype,
        )

    def _eval(self, X: Tensor) -> Tensor:
        x = X[..., :-1]
        s = X[..., -1].clamp(0.0, 1.0)
        b_s = self.b - 0.1 * (1.0 - s)
        part1 = -self.a * torch.exp(
            -b_s / math.sqrt(self.design_dim) * torch.linalg.norm(x, dim=-1)
        )
        part2 = -torch.exp(torch.mean(torch.cos(self.c * x), dim=-1))
        return part1 + part2 + self.a + math.e

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)


class AugmentedBukin(SyntheticTestFunction):
    r"""Continuous-fidelity Bukin benchmark with realistic low-fidelity distortion.

    The first two dimensions are the original Bukin design variables and the
    final dimension is a continuous fidelity parameter `s in [0, 1]`. At
    `s = 1`, this matches BoTorch's single-fidelity Bukin function exactly.

    As fidelity decreases, the surface becomes smoother, slightly warped, and
    systematically biased, which is closer to how coarse simulators behave in
    practice than a pure additive perturbation.
    """

    dim = 3
    continuous_inds = list(range(dim))
    _bounds = [(-15.0, -5.0), (-3.0, 3.0), (0.0, 1.0)]
    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0, 1.0)]
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        _init_synthetic_test_function(
            self,
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

    @staticmethod
    def _bukin_core(x1: Tensor, x2: Tensor) -> Tensor:
        return 100.0 * torch.sqrt(torch.abs(x2 - 0.01 * x1.pow(2))) + 0.01 * torch.abs(x1 + 10.0)

    def _eval(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        s = X[..., 2].clamp(0.0, 1.0)
        delta = 1.0 - s

        x1n = (x1 + 15.0) / 10.0
        x2n = (x2 + 3.0) / 6.0

        x1w = torch.clamp(
            x1 + 0.85 * delta * (1.4 * torch.sin(math.pi * x2n) - 0.55 * torch.cos(1.5 * math.pi * x1n)),
            -15.0,
            -5.0,
        )
        x2w = torch.clamp(
            x2 + 0.55 * delta * (0.8 * torch.cos(math.pi * x1n) + 0.35 * torch.sin(2.0 * math.pi * x2n)),
            -3.0,
            3.0,
        )

        ridge = x2w - (0.01 + 0.0015 * delta) * x1w.pow(2) + 0.45 * delta * (x1n - 0.45)
        ridge_smoothing = 0.03 * delta + 0.01 * delta * (1.0 - x2n)

        part1 = (100.0 - 18.0 * delta) * (ridge.pow(2) + ridge_smoothing.pow(2)).pow(0.25)
        part2 = (0.01 + 0.003 * delta) * torch.sqrt((x1w + 10.0).pow(2) + (0.35 * delta).pow(2))
        bias = delta * (
            1.8
            + 1.5 * torch.sin(math.pi * x1n) * torch.cos(0.5 * math.pi * x2n)
            + 0.9 * (x2n - 0.6)
        )
        return part1 + part2 + bias

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)


class AugmentedRastrigin(SyntheticTestFunction):
    r"""Continuous-fidelity Rastrigin benchmark with structured low-fidelity bias.

    The first `dim` coordinates are the original Rastrigin design variables and
    the final coordinate is a continuous fidelity parameter `s in [0, 1]`. At
    `s = 1`, this matches BoTorch's single-fidelity Rastrigin function exactly.

    Lower fidelities smooth the oscillatory structure, warp the inputs, and add
    a correlated discrepancy term, which mimics coarse models that retain the
    global basin but miss narrow local structure.
    """

    _optimal_value = 0.0
    _check_grad_at_opt = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        if dim < 1:
            raise ValueError("AugmentedRastrigin needs dim >= 1")
        self.design_dim = dim
        self.dim = dim + 1
        self.continuous_inds = list(range(self.dim))
        bounds = [(-5.12, 5.12) for _ in range(dim)] + [(0.0, 1.0)]
        self._bounds = bounds
        self._optimizers = [tuple([0.0 for _ in range(dim)] + [1.0])]
        _init_synthetic_test_function(
            self,
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
            dtype=dtype,
        )

    def _eval(self, X: Tensor) -> Tensor:
        x = X[..., :-1]
        s = X[..., -1].clamp(0.0, 1.0)
        delta = 1.0 - s
        delta_x = delta.unsqueeze(-1)

        xn = x / 5.12
        neighbor = torch.roll(xn, shifts=1, dims=-1)

        xw = torch.clamp(
            x + 0.55 * delta_x * (1.4 * torch.sin(math.pi * neighbor) - 0.7 * torch.cos(1.5 * math.pi * xn)),
            -5.12,
            5.12,
        )
        amplitude = 10.0 - 6.5 * delta_x
        frequency = 1.0 - 0.22 * delta_x
        quadratic_scale = 1.0 - 0.18 * delta_x
        phase = 0.15 * delta_x * torch.sin(math.pi * neighbor)

        response = amplitude[..., 0] * self.design_dim + torch.sum(
            quadratic_scale * xw.pow(2)
            - amplitude * torch.cos(2.0 * math.pi * (frequency * xw + phase)),
            dim=-1,
        )
        discrepancy = delta * (
            0.35 * torch.sum((xn - 0.35 * neighbor).pow(2), dim=-1)
            + 1.2 * torch.mean(torch.sin(math.pi * (xn + 0.5 * neighbor)), dim=-1)
        )
        return response + discrepancy

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)


class PenicillinYield(SyntheticTestFunction):
    r"""Single-objective Penicillin benchmark returning penicillin yield.

    This reuses BoTorch's built-in multi-objective Penicillin simulator and
    keeps only the yield objective. Unlike the original multi-objective
    benchmark, this class returns the penicillin yield directly, so higher is
    better by default.
    """

    dim = BoTorchPenicillin.dim
    continuous_inds = list(range(dim))
    _bounds = list(BoTorchPenicillin._bounds)
    _check_grad_at_opt = False

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        _init_synthetic_test_function(
            self,
            noise_std=noise_std,
            negate=negate,
            dtype=dtype,
        )

    def _eval(self, X: Tensor) -> Tensor:
        values = BoTorchPenicillin.penicillin_vectorized(
            X.reshape(-1, self.dim).clone()
        )
        return (-values[..., 0]).reshape(*X.shape[:-1])

    def evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return self._eval(X)



import os
import numpy as np
import torch
try:
    from olympus.emulators import Emulator
except ImportError:
    Emulator = None
tkwargs = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

class SuzukiHFEmulator:
    """
    High-fidelity Suzuki from Olympus emulator.
    Uses input format [x1,x2,x3,x4,s], but s is fixed to 1 via bounds.
    """
    def __init__(self, num_samples: int = 16, clip=(0.0, 100.0)):
        if Emulator is None:
            raise ImportError("SuzukiHFEmulator requires olympus to be installed.")
        os.environ.setdefault("OLYMPUS_SCRATCH", "/tmp/olympus_scratch")
        os.makedirs(os.environ["OLYMPUS_SCRATCH"], exist_ok=True)

        self.emu = Emulator(dataset="suzuki", model="BayesNeuralNet")
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
