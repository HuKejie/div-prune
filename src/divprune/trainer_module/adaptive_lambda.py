"""Feedback controller for adaptive divergence-regularization weighting."""

import math


class DivergenceController:
    """Integral (feedback) controller for the lambda of the divergence term.

    The controller raises lambda while head similarity exceeds the target
    diversity budget and lowers it once the budget is met:

        lambda_t = clamp(lambda_{t-1} + eta * (div_t - target), 0, lambda_max)

    Starting at lambda = 0 means the regularizer only interferes when the
    model's natural head similarity violates the budget -- in contrast to a
    fixed lambda, which is either too weak to move similarity or too strong
    and hurts the task (both failure modes observed in the fixed-lambda sweep).
    """

    def __init__(self, target: float, eta: float, lambda_max: float) -> None:
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(f"target must be finite and in [0, 1], got {target}")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError(f"eta must be finite and positive, got {eta}")
        if not math.isfinite(lambda_max) or lambda_max < 0:
            raise ValueError(f"lambda_max must be finite and >= 0, got {lambda_max}")
        self.target = target
        self.eta = eta
        self.lambda_max = lambda_max
        self.lambda_val = 0.0

    def update(self, div_t: float) -> float:
        """Update lambda from the current divergence estimate and return the new value.

        Non-finite divergence estimates leave lambda unchanged.
        """
        if not math.isfinite(div_t):
            return self.lambda_val
        self.lambda_val = max(
            0.0,
            min(self.lambda_max, self.lambda_val + self.eta * (div_t - self.target)),
        )
        return self.lambda_val
