"""Privacy attacks used for evaluation (never for tuning).

attribute_inference  logistic-regression attacker that predicts a sensitive
                     attribute of a released (sanitized) record from its other
                     sanitized coordinates (paper experiment E6).
membership           loss-threshold membership inference (Yeom et al. 2018)
                     on the final global model.
"""

from . import attribute_inference, membership  # noqa: F401
