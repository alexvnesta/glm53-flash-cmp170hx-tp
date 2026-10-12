"""Bound queued HTTP callers while preserving the existing single GPU lock.

This owns admission only. The API retains the acquired lock until its existing
shielded native cancellation/cleanup finishes. No GPU job is created here.
"""
import asyncio
import math
import os
from dataclasses import dataclass


class AdmissionError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class AdmissionPolicy:
    max_waiters: int = 8
    timeout_seconds: float = 120
    poll_seconds: float = 0.25

    def validate(self):
        if isinstance(self.max_waiters, bool) or not isinstance(self.max_waiters, int) or not 1 <= self.max_waiters <= 64:
            raise ValueError('Request queue capacity must be an integer1..64')
        if not math.isfinite(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 1800:
            raise ValueError('Request queue deadline must be1..1800 seconds')
        if not math.isfinite(self.poll_seconds) or not 0 < self.poll_seconds <= 1:
            raise ValueError('Disconnect polling interval must be positive and at most1 second')

    @classmethod
    def from_environment(cls):
        policy = cls(max_waiters=int(os.environ.get('GLM53_REQUEST_QUEUE_SIZE', '8')),
                     timeout_seconds=float(os.environ.get('GLM53_REQUEST_WAIT_SECONDS', '120')))
        policy.validate()
        return policy


class RequestAdmission:
    def __init__(self, policy=AdmissionPolicy()):
        policy.validate()
        self.policy = policy
        self.waiting = 0
        self.metrics = dict(admitted=0, rejected=0, timed_out=0, disconnected=0, cancelled=0)

    def statistics(self):
        return dict(waiting=self.waiting, max_waiters=self.policy.max_waiters,
                    timeout_seconds=self.policy.timeout_seconds, **self.metrics)

    async def acquire(self, lock, request):
        started = asyncio.get_running_loop().time()
        # Called on the HTTP loop. Reservation and counter updates do not await.
        if self.waiting >= self.policy.max_waiters:
            self.metrics['rejected'] += 1
            raise AdmissionError(429, 'Request queue is full; retry later')
        self.waiting += 1
        pending = None
        transferred = False
        try:
            if await request.is_disconnected():
                self.metrics['disconnected'] += 1
                raise AdmissionError(499, 'Client disconnected while queued')
            pending = asyncio.create_task(lock.acquire())
            deadline = asyncio.get_running_loop().time() + self.policy.timeout_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self.metrics['timed_out'] += 1
                    raise AdmissionError(429, 'Request queue deadline exceeded; retry later')
                completed, _ = await asyncio.wait({pending}, timeout=min(remaining, self.policy.poll_seconds))
                disconnected = await request.is_disconnected()
                if disconnected:
                    self.metrics['disconnected'] += 1
                    raise AdmissionError(499, 'Client disconnected while queued')
                if completed:
                    if pending.result() is not True:
                        raise RuntimeError('Request lock did not return ownership')
                    transferred = True
                    self.metrics['admitted'] += 1
                    return asyncio.get_running_loop().time() - started
        except asyncio.CancelledError:
            self.metrics['cancelled'] += 1
            raise
        finally:
            if pending is not None and not transferred:
                pending.cancel()
                # asyncio.wait did not cancel the acquire task. Resolve its
                # ownership even when cancellation races a successful acquire.
                try:
                    acquired = await pending
                except asyncio.CancelledError:
                    acquired = False
                if acquired:
                    lock.release()
            self.waiting -= 1
