"""Backpressure control for producer-consumer pipeline.

Implements a token-rate regulator with hysteresis to prevent
queue overflow in the HSSLM-S parallel pipeline.

State machine:
    RUNNING --[fill>high_watermark]--> THROTTLING --[fill<low_watermark]--> RUNNING

This prevents oscillation by requiring the fill ratio to cross
different thresholds for state transitions (hysteresis).
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto


class BackpressureState(Enum):
    """Finite states for the backpressure controller."""
    RUNNING = auto()      # Normal operation, producer can produce
    THROTTLING = auto()   # Queue filling, producer should slow/stop
    RECOVERING = auto()   # Queue draining, producer can cautiously resume


@dataclass
class BackpressureMetrics:
    """Real-time queue metrics for backpressure decisions.
    
    Attributes:
        queue_depth: Current number of items in queue
        queue_capacity: Maximum queue capacity
        fill_ratio: queue_depth / queue_capacity (0.0 to 1.0)
        producer_rate: Items produced per second
        consumer_rate: Items consumed per second
        state: Current backpressure state
        timestamp: When metrics were recorded
    """
    queue_depth: int = 0
    queue_capacity: int = 1
    fill_ratio: float = 0.0
    producer_rate: float = 0.0      # items/sec
    consumer_rate: float = 0.0      # items/sec
    state: BackpressureState = BackpressureState.RUNNING
    timestamp: float = field(default_factory=time.monotonic)
    
    @property
    def net_rate(self) -> float:
        """Net production rate (positive = growing, negative = shrinking)."""
        return self.producer_rate - self.consumer_rate
    
    def should_throttle(self, high_watermark: float = 0.8) -> bool:
        """Check if throttling should begin.
        
        Args:
            high_watermark: Fill ratio threshold to start throttling
        
        Returns:
            True if fill ratio exceeds high watermark
        """
        return self.fill_ratio > high_watermark
    
    def should_resume(self, low_watermark: float = 0.3) -> bool:
        """Check if production can resume.
        
        Args:
            low_watermark: Fill ratio threshold to resume production
        
        Returns:
            True if fill ratio has dropped below low watermark
        """
        return self.fill_ratio < low_watermark
    
    def is_critical(self, critical_watermark: float = 0.95) -> bool:
        """Check if queue is critically full.
        
        Args:
            critical_watermark: Critical fill ratio
        
        Returns:
            True if queue is near-overflow
        """
        return self.fill_ratio > critical_watermark


class BackpressureController:
    """Prevents queue overflow with hysteresis.
    
    Uses a state machine with three states (RUNNING, THROTTLING,
    RECOVERING) and different thresholds for transitions to prevent
    rapid oscillation (hysteresis).
    
    Typical usage:
        controller = BackpressureController(high_watermark=0.8, low_watermark=0.3)
        
        # In producer loop:
        metrics = controller.update(queue_depth=current_depth, capacity=max_capacity)
        if controller.should_produce():
            # Safe to produce
            produce_items()
            controller.record_produced(n_items)
        else:
            # Wait or apply backpressure
            time.sleep(0.001)
    
    Args:
        high_watermark: Fill ratio to start throttling (default 0.8)
        low_watermark: Fill ratio to resume production (default 0.3)
        critical_watermark: Fill ratio for emergency stop (default 0.95)
        adaptation_rate: How quickly to adapt rate limits (0=none, 1=instant)
        sample_window: Time window for rate averaging in seconds (default 1.0)
    """
    
    def __init__(self, high_watermark: float = 0.8, low_watermark: float = 0.3,
                 critical_watermark: float = 0.95, adaptation_rate: float = 0.1,
                 sample_window: float = 1.0):
        """Initialize backpressure controller.
        
        Args:
            high_watermark: Fill ratio to start throttling
            low_watermark: Fill ratio to resume production
            critical_watermark: Fill ratio for emergency stop
            adaptation_rate: Rate limit adaptation speed
            sample_window: Time window for rate measurement
        """
        if not (0 < low_watermark < high_watermark < critical_watermark <= 1.0):
            raise ValueError(
                f"Invalid watermarks: must satisfy "
                f"0 < low ({low_watermark}) < high ({high_watermark}) < "
                f"critical ({critical_watermark}) <= 1.0"
            )
        
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.critical_watermark = critical_watermark
        self.adaptation_rate = adaptation_rate
        self.sample_window = sample_window
        
        self._state = BackpressureState.RUNNING
        
        # Rate tracking
        self._tokens_produced = 0
        self._tokens_consumed = 0
        self._last_check_time = time.monotonic()
        
        # Adaptive rate limit (1.0 = full speed, 0.0 = stopped)
        self._rate_limit = 1.0
        
        # Statistics
        self._throttle_events = 0
        self._total_tokens_produced = 0
        self._total_tokens_consumed = 0
        self._total_throttle_time = 0.0
    
    def update(self, queue_depth: int, queue_capacity: int,
               force_check: bool = False) -> BackpressureMetrics:
        """Update controller state with current queue metrics.
        
        Args:
            queue_depth: Current number of items in queue
            queue_capacity: Maximum queue capacity
            force_check: Force state recalculation even if sample window
                        hasn't elapsed
        
        Returns:
            BackpressureMetrics with current state
        """
        now = time.monotonic()
        dt = now - self._last_check_time
        
        # Calculate fill ratio
        fill_ratio = queue_depth / max(queue_capacity, 1)
        
        # Calculate rates if sample window elapsed or forced
        if dt >= self.sample_window or force_check:
            producer_rate = self._tokens_produced / max(dt, 1e-6)
            consumer_rate = self._tokens_consumed / max(dt, 1e-6)
            
            # Reset counters
            self._tokens_produced = 0
            self._tokens_consumed = 0
            self._last_check_time = now
        else:
            # Use previous rates
            if hasattr(self, '_prev_metrics'):
                producer_rate = self._prev_metrics.producer_rate
                consumer_rate = self._prev_metrics.consumer_rate
            else:
                producer_rate = 0.0
                consumer_rate = 0.0
        
        # State machine with hysteresis
        if self._state == BackpressureState.RUNNING:
            if fill_ratio > self.critical_watermark:
                self._state = BackpressureState.THROTTLING
                self._throttle_events += 1
                self._rate_limit = 0.0  # Emergency stop
            elif fill_ratio > self.high_watermark:
                self._state = BackpressureState.THROTTLING
                self._throttle_events += 1
                # Gradually reduce rate
                self._rate_limit *= (1.0 - self.adaptation_rate)
        
        elif self._state == BackpressureState.THROTTLING:
            if fill_ratio < self.low_watermark:
                self._state = BackpressureState.RUNNING
                self._rate_limit = min(1.0, self._rate_limit + self.adaptation_rate)
            elif fill_ratio < self.high_watermark * 0.9:
                self._state = BackpressureState.RECOVERING
                self._rate_limit = min(0.5, self._rate_limit + self.adaptation_rate * 0.5)
            else:
                # Still throttling, reduce rate further
                self._rate_limit *= (1.0 - self.adaptation_rate)
                self._total_throttle_time += dt
        
        elif self._state == BackpressureState.RECOVERING:
            if fill_ratio < self.low_watermark:
                self._state = BackpressureState.RUNNING
                self._rate_limit = min(1.0, self._rate_limit + self.adaptation_rate)
            elif fill_ratio > self.high_watermark:
                self._state = BackpressureState.THROTTLING
                self._rate_limit *= (1.0 - self.adaptation_rate)
            else:
                # Gradually increase rate while recovering
                self._rate_limit = min(0.8, self._rate_limit + self.adaptation_rate * 0.3)
        
        # Clamp rate limit
        self._rate_limit = max(0.0, min(1.0, self._rate_limit))
        
        metrics = BackpressureMetrics(
            queue_depth=queue_depth,
            queue_capacity=queue_capacity,
            fill_ratio=fill_ratio,
            producer_rate=producer_rate,
            consumer_rate=consumer_rate,
            state=self._state,
            timestamp=now
        )
        
        self._prev_metrics = metrics
        return metrics
    
    def should_produce(self) -> bool:
        """Check if producer should produce.
        
        Returns:
            True if production is allowed (not throttling)
        """
        return self._state != BackpressureState.THROTTLING
    
    def should_consume(self) -> bool:
        """Check if consumer should consume.
        
        Consumer always runs, but this can be used to signal
        priority consumption when queue is full.
        
        Returns:
            True (consumer always runs)
        """
        return True
    
    def record_produced(self, n: int = 1):
        """Record that items were produced.
        
        Args:
            n: Number of items produced
        """
        self._tokens_produced += n
        self._total_tokens_produced += n
    
    def record_consumed(self, n: int = 1):
        """Record that items were consumed.
        
        Args:
            n: Number of items consumed
        """
        self._tokens_consumed += n
        self._total_tokens_consumed += n
    
    def get_rate_limit(self) -> float:
        """Get current adaptive rate limit.
        
        Returns:
            Float in [0.0, 1.0] where 1.0 = full speed, 0.0 = stopped
        """
        return self._rate_limit
    
    def get_metrics(self) -> BackpressureMetrics:
        """Get the most recent metrics.
        
        Returns:
            Cached BackpressureMetrics
        """
        if hasattr(self, '_prev_metrics'):
            return self._prev_metrics
        return BackpressureMetrics(state=self._state)
    
    def get_stats(self) -> dict:
        """Get cumulative statistics.
        
        Returns:
            Dictionary with throttle events, total tokens, etc.
        """
        return {
            'throttle_events': self._throttle_events,
            'total_produced': self._total_tokens_produced,
            'total_consumed': self._total_tokens_consumed,
            'total_throttle_time_sec': self._total_throttle_time,
            'current_state': self._state.name,
            'current_rate_limit': self._rate_limit,
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self._state = BackpressureState.RUNNING
        self._tokens_produced = 0
        self._tokens_consumed = 0
        self._last_check_time = time.monotonic()
        self._rate_limit = 1.0


class AdaptiveBackpressure(BackpressureController):
    """Backpressure controller with adaptive watermark tuning.
    
    Automatically adjusts high and low watermarks based on observed
    throughput patterns to optimize for different workload types.
    
    For bursty workloads: higher watermarks
    For steady workloads: lower watermarks
    
    Args:
        base_high: Starting high watermark (default 0.8)
        base_low: Starting low watermark (default 0.3)
        adaptation_rate: How quickly to adapt watermarks
        target_utilization: Target queue utilization (default 0.6)
    """
    
    def __init__(self, base_high: float = 0.8, base_low: float = 0.3,
                 adaptation_rate: float = 0.05, target_utilization: float = 0.6):
        super().__init__(
            high_watermark=base_high,
            low_watermark=base_low,
            adaptation_rate=adaptation_rate
        )
        self._base_high = base_high
        self._base_low = base_low
        self._target_utilization = target_utilization
        self._utilization_history = []
        self._history_max = 10
    
    def update(self, queue_depth: int, queue_capacity: int,
               force_check: bool = False) -> BackpressureMetrics:
        """Update with adaptive watermark tuning.
        
        Overrides parent to add watermark adaptation based on
        observed utilization patterns.
        """
        metrics = super().update(queue_depth, queue_capacity, force_check)
        
        # Track utilization for adaptive tuning
        self._utilization_history.append(metrics.fill_ratio)
        if len(self._utilization_history) > self._history_max:
            self._utilization_history.pop(0)
        
        # Adapt watermarks if we have enough history
        if len(self._utilization_history) >= 3:
            avg_util = sum(self._utilization_history) / len(self._utilization_history)
            
            if avg_util > self._target_utilization + 0.1:
                # Running too full: raise high watermark
                self.high_watermark = min(0.95, self.high_watermark + self.adaptation_rate)
                self.low_watermark = min(self.high_watermark - 0.1,
                                         self.low_watermark + self.adaptation_rate)
            elif avg_util < self._target_utilization - 0.1:
                # Running too empty: lower watermarks
                self.high_watermark = max(0.5, self.high_watermark - self.adaptation_rate)
                self.low_watermark = max(0.1, self.low_watermark - self.adaptation_rate)
        
        return metrics
