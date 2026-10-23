"""Choose a working inference device once at application startup."""

import logging

logger = logging.getLogger(__name__)


def resolve_prediction_device(requested: str = "auto") -> str:
    """Prefer usable CUDA in auto mode; preserve explicit device requests."""
    import torch

    requested = requested.strip().lower()
    if requested != "auto":
        device = str(torch.device(requested))
        logger.info("Prediction device: %s (explicit)", device)
        return device

    if torch.cuda.is_available():
        try:
            # Availability alone does not establish driver/kernel compatibility.
            probe = torch.ones((2, 2), device="cuda")
            torch.mm(probe, probe)
            torch.cuda.synchronize()
        except (RuntimeError, AssertionError) as exc:
            logger.warning("CUDA inference probe failed; using CPU: %s", exc)
        else:
            logger.info("Prediction device: cuda (%s)", torch.cuda.get_device_name(0))
            return "cuda"

    logger.info("Prediction device: cpu (CUDA unavailable or unusable)")
    return "cpu"
