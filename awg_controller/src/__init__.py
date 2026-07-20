"""AWG controller source modules.

Re-exports the public API from ``awg_control`` and ``dds_strategies``
so users can write::

    from awg_controller.src import RFConverter, AWGBatch, DDSRampStrategy
"""

from awg_controller.src.awg_control import (
    AODSettings,
    AWGBatch,
    RFConverter,
    RFRamp,
    compute_core_assignments,
    validate_hardware_limits,
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    MIN_MOVE_DURATION_S,
    TOTAL_DDS_CORES,
)
from awg_controller.src.camera import (
    Camera,
    RealArrayCamera,
)
from awg_controller.src.offline_camera import (
    GaussianCameraConfig,
    OfflineArrayCamera,
)
from awg_controller.src.dds_strategies import (
    DDSStrategy,
    DDSStreamingStrategy,
    DDSRampStrategy,
    DDSPatternStrategy,
    DDSCameraTriggeredStrategy,
    RampConfig,
    PatternConfig,
    CameraTriggerConfig,
    STRATEGY_REGISTRY,
    get_strategy,
    transport_duration_s,
    wait_transport,
    prefill_count_for_timer,
)
from awg_controller.src.scapp_gen import (
    ScappFeeder,
    ScappFeederConfig,
    ToneSegment,
    segment_instantaneous_phase,
)

__all__ = [
    # awg_control
    "AODSettings",
    "AWGBatch",
    "RFConverter",
    "RFRamp",
    "compute_core_assignments",
    "validate_hardware_limits",
    "MAX_AMPLITUDE_PCT_PER_CHANNEL",
    "MIN_MOVE_DURATION_S",
    "TOTAL_DDS_CORES",
    # camera
    "Camera",
    "RealArrayCamera",
    # offline_camera
    "GaussianCameraConfig",
    "OfflineArrayCamera",
    # dds_strategies
    "DDSStrategy",
    "DDSStreamingStrategy",
    "DDSRampStrategy",
    "DDSPatternStrategy",
    "DDSCameraTriggeredStrategy",
    "RampConfig",
    "PatternConfig",
    "CameraTriggerConfig",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "transport_duration_s",
    "wait_transport",
    "prefill_count_for_timer",
    # scapp_gen
    "ScappFeeder",
    "ScappFeederConfig",
    "ToneSegment",
    "segment_instantaneous_phase",
]
