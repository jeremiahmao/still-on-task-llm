"""Sliding-window rate limiter for API calls (tokens + requests per minute)."""

import asyncio
import time
from collections import deque


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


class AsyncRateLimiter:
    """Dual sliding-window rate limiter (TPM + RPM) for async API calls.

    Cerebras enforces both tokens-per-minute and requests-per-minute.
    Paid tier for qwen-3-235b: 500K TPM, 500 RPM.
    Free tier: 60K TPM, 30 RPM.
    """

    def __init__(self, tokens_per_minute: int = 500_000, requests_per_minute: int = 500):
        self.tpm = tokens_per_minute
        self.rpm = requests_per_minute
        self._lock = asyncio.Lock()
        self._token_window: deque[tuple[float, int]] = deque()
        self._request_window: deque[float] = deque()

    async def acquire(self, estimated_tokens: int):
        """Wait until both token and request budgets allow, then reserve."""
        async with self._lock:
            while True:
                now = time.monotonic()
                # Prune entries older than 60s
                while self._token_window and now - self._token_window[0][0] >= 60:
                    self._token_window.popleft()
                while self._request_window and now - self._request_window[0] >= 60:
                    self._request_window.popleft()

                tokens_used = sum(n for _, n in self._token_window)
                requests_used = len(self._request_window)

                token_ok = tokens_used + estimated_tokens <= self.tpm
                request_ok = requests_used + 1 <= self.rpm

                if token_ok and request_ok:
                    break

                # Figure out how long to wait
                waits = []
                if not token_ok and self._token_window:
                    waits.append(60 - (now - self._token_window[0][0]))
                if not request_ok and self._request_window:
                    waits.append(60 - (now - self._request_window[0]))
                wait = max(min(waits) if waits else 0.1, 0.05)

                # Release lock while sleeping so other coroutines aren't blocked
                self._lock.release()
                await asyncio.sleep(wait)
                await self._lock.acquire()

            self._token_window.append((time.monotonic(), estimated_tokens))
            self._request_window.append(time.monotonic())


class SyncRateLimiter:
    """Dual sliding-window rate limiter (TPM + RPM) for synchronous API calls."""

    def __init__(self, tokens_per_minute: int = 500_000, requests_per_minute: int = 500):
        self.tpm = tokens_per_minute
        self.rpm = requests_per_minute
        self._token_window: deque[tuple[float, int]] = deque()
        self._request_window: deque[float] = deque()

    def acquire(self, estimated_tokens: int):
        """Wait until both token and request budgets allow, then reserve."""
        while True:
            now = time.monotonic()
            while self._token_window and now - self._token_window[0][0] >= 60:
                self._token_window.popleft()
            while self._request_window and now - self._request_window[0] >= 60:
                self._request_window.popleft()

            tokens_used = sum(n for _, n in self._token_window)
            requests_used = len(self._request_window)

            token_ok = tokens_used + estimated_tokens <= self.tpm
            request_ok = requests_used + 1 <= self.rpm

            if token_ok and request_ok:
                break

            waits = []
            if not token_ok and self._token_window:
                waits.append(60 - (now - self._token_window[0][0]))
            if not request_ok and self._request_window:
                waits.append(60 - (now - self._request_window[0]))
            time.sleep(max(min(waits) if waits else 0.1, 0.05))

        self._token_window.append((time.monotonic(), estimated_tokens))
        self._request_window.append(time.monotonic())


# Keep old names as aliases for backward compat within this session
AsyncTokenRateLimiter = AsyncRateLimiter
SyncTokenRateLimiter = SyncRateLimiter
