# SynIOL Dataset

## Purpose

SynIOL (Synthetic Industrial Operational Logs) is generated from latent industrial behavior dynamics. It is designed to evaluate whether log representation methods can recover behavior structure from observable logs without directly seeing latent labels.

## Model Visibility

Use `syniol_observable.csv` as model input. Hidden/evaluation fields are stored separately in `syniol_labels.csv` and must not be used as model input. `syniol_full.csv` is provided only for audit, debugging, and convenient analysis.

## Generated Size

- Events: 76800
- Trajectories: 120
- Events per trajectory: 640
- Transition events: 9157
- Seed: 42

## Output Files

- `syniol_observable.csv`: Observable fields only; use this file as model input.
- `syniol_labels.csv`: Hidden latent/evaluation fields; do not use this file as model input.
- `syniol_full.csv`: Convenience file containing both observable fields and hidden labels.
- `syniol_manifest.json`: Machine-readable generation metadata.
- `syniol_description.md`: Human-readable dataset description.

## Observable Columns

- trajectory_id
- event_index
- timestamp
- event_type
- component
- severity
- state
- message
- sensor_value
- control_value

## Hidden / Evaluation Columns

- trajectory_id
- event_index
- segment_id
- behavior_id
- behavior_name
- variant_id
- variant_name
- is_transition
- source_behavior_id
- target_behavior_id
- transition_progress
- risk_level
- load_level
- recovery_progress
- component_health_sensor
- component_health_control
- component_health_network
- component_health_processing
- primary_component

## Behavior Catalog

- 0: Fast Stable Recovery (fast_recovery_a, fast_recovery_b, fast_recovery_c)
- 1: Delayed Recovery (delayed_recovery_a, delayed_recovery_b, delayed_recovery_c)
- 2: Oscillating Instability (oscillating_instability_a, oscillating_instability_b, oscillating_instability_c)
- 3: Cascading Failure (cascading_failure_a, cascading_failure_b, cascading_failure_c)
- 4: Normal Stable Operation (normal_stable_a, normal_stable_b, normal_stable_c)

## Event Vocabulary

- STATUS_CHECK: strongest components: SensorUnit, ControlUnit
- VALUE_READ: strongest components: SensorUnit, ProcessingUnit
- SETPOINT_UPDATE: strongest components: ControlUnit, ProcessingUnit
- CONTROL_COMMAND: strongest components: ControlUnit, ProcessingUnit
- ACTUATOR_RESPONSE: strongest components: ControlUnit, ProcessingUnit
- WARNING_RAISED: strongest components: SensorUnit, ControlUnit
- ERROR_DETECTED: strongest components: SensorUnit, ControlUnit
- COMM_DELAY: strongest components: NetworkUnit, ProcessingUnit
- NETWORK_TIMEOUT: strongest components: NetworkUnit, ProcessingUnit
- DIAGNOSIS_START: strongest components: ProcessingUnit, ControlUnit
- RECOVERY_ACTION: strongest components: ControlUnit, ProcessingUnit
- RETRY_OPERATION: strongest components: ControlUnit, NetworkUnit
- MAINTENANCE_CHECK: strongest components: SensorUnit, ControlUnit
- HEALTH_OK: strongest components: SensorUnit, ControlUnit

## Notes

- Event type and severity are intentionally not perfectly bound; severity is risk-based with event-aware soft constraints.
- Message text uses template variation to avoid a single rigid message per event type.
- Component-event dependencies are encoded through component weights, so events such as `VALUE_READ`, `RECOVERY_ACTION`, and `NETWORK_TIMEOUT` naturally favor different components.
- `Normal Stable Operation` is included to make the dataset closer to real operational logs, where many intervals are normal rather than fault-recovery dominated.
