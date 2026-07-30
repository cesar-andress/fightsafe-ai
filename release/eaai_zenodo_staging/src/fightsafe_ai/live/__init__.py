"""Live / pseudo-stream processing from local video (preview + event bus; demo hooks)."""

from fightsafe_ai.live.event_bus import (
    EventBus,
    EventCategory,
    SafetyEvent,
    SafetyLevel,
    SeverityNormalization,
    normalize_level_from_score,
)
from fightsafe_ai.live.gpu_monitor import get_nvidia_gpu_metrics, shutdown_gpu_monitor
from fightsafe_ai.live.live_overlay import draw_live_overlay
from fightsafe_ai.live.live_pipeline import LiveFrameResult, LivePipeline, LivePipelineConfig
from fightsafe_ai.live.performance import LivePerformanceMonitor, PerformanceSnapshot
from fightsafe_ai.live.video_source import (
    FileVideoSource,
    VideoFrameMeta,
    VideoSource,
    WebcamSource,
    open_video_source,
)


__all__ = [
    "EventBus",
    "EventCategory",
    "FileVideoSource",
    "LiveFrameResult",
    "LivePerformanceMonitor",
    "LivePipeline",
    "LivePipelineConfig",
    "PerformanceSnapshot",
    "SafetyEvent",
    "SafetyLevel",
    "SeverityNormalization",
    "VideoFrameMeta",
    "VideoSource",
    "WebcamSource",
    "draw_live_overlay",
    "get_nvidia_gpu_metrics",
    "normalize_level_from_score",
    "open_video_source",
    "shutdown_gpu_monitor",
]
