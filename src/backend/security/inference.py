"""Keep prediction admission separate from model/scaler and pipeline identity."""

from contextlib import contextmanager

from fastapi import HTTPException, Request


@contextmanager
def prediction_slot(request: Request):
    """Reserve one process-wide prediction slot, rejecting excess work immediately.

    Args:
        request: Request whose app owns a bounded prediction semaphore.

    Yields:
        None. The slot covers model loading, input preparation, inference and saving.
        It is held until the synchronous work finishes, including after client timeout.

    Raises:
        HTTPException: 503 INFERENCE_BUSY with Retry-After when all slots are in use.
    """
    slots = request.app.state.prediction_slots
    if not slots.acquire(blocking=False):
        raise HTTPException(
            503,
            {
                "code": "INFERENCE_BUSY",
                "message": "Prediction capacity is busy. Try again shortly.",
            },
            headers={"Retry-After": "1"},
        )
    try:
        yield
    finally:
        slots.release()
