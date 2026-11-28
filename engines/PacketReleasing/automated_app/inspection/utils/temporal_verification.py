"""Temporal verification utilities for consecutive-frame confirmation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class TemporalDecision(Generic[T]):
    raw_value: T
    confirmed_value: Optional[T]
    is_confirmed: bool
    streak: int
    required_frames: int
    window_agreement: float


class TemporalVerifier(Generic[T]):
    """Requires consecutive stable observations before confirming a discrete state.

    Any `reset_value` (such as `UNKNOWN`) immediately resets the streak so
    unreliable frames can never accidentally confirm a state like `CLOSED`.
    """

    def __init__(
        self,
        required_frames: int = 5,
        window_size: int = 7,
        reset_values: Optional[set[T]] = None,
    ) -> None:
        self.required_frames = max(1, int(required_frames))
        self.window_size = max(self.required_frames, int(window_size))
        self.reset_values: set[T] = set(reset_values) if reset_values else set()
        self._history: Deque[T] = deque(maxlen=self.window_size)
        self._streak_value: Optional[T] = None
        self._streak_count: int = 0
        self._last_confirmed: Optional[T] = None

    def reset(self) -> None:
        self._history.clear()
        self._streak_value = None
        self._streak_count = 0
        self._last_confirmed = None

    @property
    def streak(self) -> int:
        return self._streak_count

    @property
    def last_confirmed(self) -> Optional[T]:
        return self._last_confirmed

    def update(self, value: T) -> TemporalDecision[T]:
        self._history.append(value)

        if value in self.reset_values:
            self._streak_value = None
            self._streak_count = 0
            agreement = self._history.count(value) / float(len(self._history))
            return TemporalDecision(
                raw_value=value,
                confirmed_value=None,
                is_confirmed=False,
                streak=0,
                required_frames=self.required_frames,
                window_agreement=agreement,
            )

        if value == self._streak_value:
            self._streak_count += 1
        else:
            self._streak_value = value
            self._streak_count = 1

        agreement = self._history.count(value) / float(len(self._history))
        is_confirmed = self._streak_count >= self.required_frames
        if is_confirmed:
            self._last_confirmed = value

        return TemporalDecision(
            raw_value=value,
            confirmed_value=value if is_confirmed else None,
            is_confirmed=is_confirmed,
            streak=self._streak_count,
            required_frames=self.required_frames,
            window_agreement=agreement,
        )

