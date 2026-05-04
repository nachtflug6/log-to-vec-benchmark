# Behavior Log V4

This document describes the design of the `v4` behavior-log dataset family used in the `behavior_log` experiment pipeline.

The goal of `v4` is to move away from template-like synthetic logs and toward a more realistic, controllable, behavior-driven benchmark in which:

- the label is defined by latent dynamics rather than a fixed event template
- event vocabulary is largely shared across behavior classes
- observed fields are noisy, delayed, and only partially faithful views of the underlying process
- transition regions are real parts of the data rather than placeholder flags
- hidden metadata is preserved for richer downstream evaluation

## High-Level Structure

The generator is organized in three conceptual layers:

1. latent process
2. observable emission
3. trajectory composition

These layers are intentionally separated.

The latent process defines what is really happening inside the system.

The observable emission layer defines how the hidden process becomes a log event.

The trajectory composition layer defines how multiple behavior segments and transitions are assembled into full trajectories and later turned into windows.

## Behavior Set

`v4` contains four behavior classes:

| Behavior ID | Behavior Name | Core Dynamic Pattern |
|---|---|---|
| `0` | `Fast Stable Recovery` | Risk rises briefly, then decreases quickly. Recovery progress improves fast and affected components return to healthy operation. |
| `1` | `Delayed Recovery` | Risk decreases slowly. Recovery progress improves gradually and may pause. The system can still recover, but the process is prolonged. |
| `2` | `Oscillating Instability` | Risk repeatedly rises and falls. Recovery appears to work temporarily but instability returns. |
| `3` | `Cascading Failure` | One component degrades first and the issue spreads to other components. Risk stays high or increases over time. |

These classes are not defined by a fixed sequence template. They are defined by the shape of hidden variable trajectories.

## Latent Process Specification

At each event step, the generator maintains a hidden latent state.

### Latent Variables

The main latent variables are:

| Variable | Meaning | Range |
|---|---|---|
| `risk_level_t` | Current system abnormality or risk level. Higher means less healthy. | `[0, 1]` |
| `load_level_t` | Current operating load or workload intensity. Higher load makes recovery more difficult. | `[0, 1]` |
| `recovery_progress_t` | Current progress toward recovery. Higher means closer to recovery completion. | `[0, 1]` |
| `component_health_t` | Health vector for each component. Higher means healthier. | `[0, 1]` per component |

`component_health_t` is represented as a vector:

```text
component_health_t = [
  health_sensor_t,
  health_control_t,
  health_network_t,
  health_processing_t
]
```

All latent variables are clipped to `[0, 1]` after each update step.

### Components

The hidden process tracks four components:

| Component | Meaning |
|---|---|
| `SensorUnit` | Sensor subsystem |
| `ControlUnit` | Control or actuator subsystem |
| `NetworkUnit` | Communication subsystem |
| `ProcessingUnit` | Internal processing or decision subsystem |

### Shared Update Logic

All behaviors share the same high-level update idea:

- `risk_level` changes according to damage, load pressure, recovery effect, and noise
- `recovery_progress` changes according to recovery speed, regression, and noise
- `load_level` evolves as a slow nuisance-factor random walk
- `component_health` changes according to damage, recovery, fluctuation, or propagation

Conceptually:

```text
risk_level_{t+1}
= current risk
+ new damage
+ load pressure
- recovery effect
+ random noise

recovery_progress_{t+1}
= current recovery progress
+ recovery gain
- regression
+ random noise

load_level_{t+1}
= slow random walk around segment-level load

component_health_{t+1}
= current health
- component damage
+ component recovery
+ random noise
```

### Load Level as a Nuisance Factor

`load_level` is intentionally not the behavior label.

It affects:

- how hard recovery is
- how noisy the process becomes
- how much numeric values fluctuate

But behavior should not be directly recoverable from average load alone.

Examples:

- higher load can make recovery slower
- higher load can increase risk variability
- lower load can make the system look more stable

### Behavior-Specific Latent Dynamics

#### Behavior 0: Fast Stable Recovery

This behavior represents a system that experiences a short abnormality but recovers quickly and effectively.

Expected latent pattern:

- `risk_level` rises early and then drops quickly
- `recovery_progress` grows quickly
- the affected component's health improves quickly
- other components stay mostly stable

Core interpretation:

```text
quick risk decrease
+ quick recovery progress
+ rapid component repair
```

#### Behavior 1: Delayed Recovery

This behavior still recovers, but more slowly.

Expected latent pattern:

- `risk_level` decreases slowly
- `recovery_progress` grows slowly and may pause
- the affected component improves gradually rather than immediately
- warning-like conditions persist longer

Core interpretation:

```text
similar outcome to fast recovery
but with a much longer and less stable path
```

This makes `Fast Stable Recovery` and `Delayed Recovery` a deliberate hard pair.

#### Behavior 2: Oscillating Instability

This behavior captures repeated instability rather than steady improvement.

Expected latent pattern:

- `risk_level` rises and falls repeatedly
- `recovery_progress` improves and then partially regresses
- the same component can improve temporarily and then degrade again
- the final event is not necessarily diagnostic by itself

Core interpretation:

```text
non-monotonic risk
+ non-monotonic recovery
+ repeated return of instability
```

#### Behavior 3: Cascading Failure

This behavior captures cross-component spread.

Expected latent pattern:

- one component degrades first
- degradation propagates to one or more additional components
- `risk_level` stays high or increases
- `recovery_progress` is weak or repeatedly disrupted
- component order matters

Core interpretation:

```text
cross-component degradation spread
```

This behavior is intentionally designed so that distinguishing it from `Oscillating Instability` requires understanding whether the issue stays local or spreads across components.

### Variants

`variant_id` in `v4` is not a template variant. It is a latent-process variant.

Variants control hidden process parameters such as:

- initial shock size
- recovery speed
- pause probability
- oscillation amplitude
- oscillation period
- cascade spread path
- cascade spread speed

This keeps variation inside the process itself rather than only in emitted event order.

## Observable Emission Specification

The observable layer converts latent state into raw log events.

The key principle is:

> observed fields are noisy and partial reflections of the latent process, not exact labels.

### Event Vocabulary

`v4` uses a mostly shared event vocabulary:

| Event Type | Meaning |
|---|---|
| `STATUS_CHECK` | General system status check |
| `VALUE_READ` | Sensor or component value read |
| `ERROR_DETECTED` | Abnormal condition detected |
| `WARNING_RAISED` | Warning raised |
| `DIAGNOSIS_START` | Diagnosis process started |
| `RECOVERY_ACTION` | Recovery or control action applied |
| `RETRY_OPERATION` | Retry performed |
| `HEALTH_OK` | Healthy status reported |
| `COMM_DELAY` | Communication delay observed |
| `ESCALATION` | Issue becomes more serious or spreads |

All four behavior classes can emit most of these event types.

The main difference is supposed to come from:

- order
- timing
- repetition
- interaction with components
- interaction with latent risk and recovery

### Shared Base Event Distribution

`v4` uses a shared base event distribution and then applies only modest latent-conditioned adjustments.

The intended default shape is approximately:

| Event Type | Base Probability |
|---|---:|
| `STATUS_CHECK` | `0.15` |
| `VALUE_READ` | `0.15` |
| `ERROR_DETECTED` | `0.12` |
| `WARNING_RAISED` | `0.10` |
| `DIAGNOSIS_START` | `0.10` |
| `RECOVERY_ACTION` | `0.12` |
| `RETRY_OPERATION` | `0.10` |
| `HEALTH_OK` | `0.10` |
| `COMM_DELAY` | `0.08` |
| `ESCALATION` | `0.08` |

Then the generator adjusts this base distribution using:

- `risk_level`
- `recovery_progress`
- `component_health`
- local phase inside the segment
- whether the current event belongs to a transition region

This is meant to reduce direct event-emission shortcut leakage from `behavior_id`.

### Component Emission

Component is not uniquely determined by event type.

Instead, component is sampled from weighted choices depending on:

- event type
- the currently most affected component
- low-health components
- transition status

This means a given event type may be emitted by different components in different contexts.

That allows the model to use:

```text
what happened
+ where it happened
+ in what order
```

instead of just token identity.

### State Emission

Observed `state` in `v4` is intentionally weakened relative to the latent process.

State is no longer a direct deterministic mapping from current risk and recovery.

Instead, the generator introduces:

- state delay
- state noise
- state persistence
- coarse updates

The design idea is:

- current event state may reflect the latent condition from a few steps ago
- not every event updates the visible state
- some visible states are incorrectly recorded
- `WARNING`, `RECOVERY`, and `DEGRADED` are not perfectly synchronized with the underlying latent state

Current design knobs include:

- `state_delay_steps`
- `state_noise_prob`
- `state_update_prob`

This makes state behave more like an imperfect operational status tag.

### Severity Emission

Observed `severity` is also intentionally weakened.

Instead of behaving like an exact discretized risk bin, severity is emitted as a noisy alarm level.

The generator uses:

- delayed risk reference
- noisy thresholds
- false positives
- false negatives
- a cap on how often `CRITICAL` can appear

This prevents shortcuts such as:

```text
many CRITICAL events -> probably cascading failure
```

### Numeric Fields

`v4` keeps two numeric fields:

| Numeric Field | Meaning |
|---|---|
| `sensor_value` | Observable measurement related to risk, load, and component condition |
| `control_value` | Control effort or recovery strength |

The current implementation ties them to latent state rather than directly to labels:

- `sensor_value` is driven by `risk_level`, `load_level`, component bias, and noise
- `control_value` is driven by recovery effort and retry/recovery activity

In the current `v4` implementation, numeric values are meaningful but still relatively structured. This is one reason simple baselines such as PCA and summary statistics remain strong.

## Trajectory Composition Specification

### Multi-Segment Trajectories

A trajectory is not a single repeated cycle.

Instead, each trajectory contains multiple behavior segments separated by transition regions.

A conceptual example:

```text
trajectory_0:
segment_0 -> Fast Stable Recovery
transition
segment_1 -> Oscillating Instability
transition
segment_2 -> Delayed Recovery
```

### Segment Metadata

Each segment has hidden metadata such as:

- `segment_id`
- `behavior_id`
- `behavior_name`
- `variant_id`
- `variant_name`
- `initial_component`
- `segment_length`
- segment-level load anchor

### Transition Regions

Transitions in `v4` are real data, not just flags.

During a transition:

- latent variables are blended between source and target behaviors
- event emission is mixed and ambiguous
- state may reflect a lagged or partial transition state
- source and target behaviors are both recorded in metadata

The transition metadata includes:

- `is_transition`
- `source_behavior_id`
- `target_behavior_id`
- `transition_progress`

This makes transition windows meaningful for downstream tasks.

### Noise

`v4` includes moderate noise that does not change the label but makes the sequence less clean.

Current noise sources include:

- irrelevant event noise
- state noise
- severity noise
- numeric noise
- occasional component noise

The intended role of noise is to reduce shortcut reliance without destroying the structure of the latent process.

## Windowing and Labels

The `v4` raw dataset is later converted to windows using:

- `window_length = 20`
- `stride = 10`
- trajectory-level train/val/test splits

Current artifacts are stored under:

- raw: `artifacts/datasets/latent_behavior_sequence_v4/raw`
- preprocessed: `artifacts/datasets/latent_behavior_sequence_v4/preprocessed`
- windows: `artifacts/datasets/latent_behavior_sequence_v4/windows`

### Window Labels

The current window label is still based on dominant behavior inside the window:

- `window_behavior_id`
- `is_transition_window`

The intended design is:

- use the dominant behavior as the window label
- mark the window as transition if it overlaps multiple behaviors or transition events

This supports both:

- standard classification and retrieval
- separate analysis of clean and transition windows

## Hidden Metadata Preserved in Raw Logs

The raw `v4` logs preserve evaluation-facing metadata that is not meant to be used directly for model training.

Examples include:

- `behavior_id`
- `behavior_name`
- `segment_id`
- `variant_id`
- `variant_name`
- `is_transition`
- `source_behavior_id`
- `target_behavior_id`
- `transition_progress`
- `risk_level`
- `load_level`
- `recovery_progress`
- `component_health_sensor`
- `component_health_control`
- `component_health_network`
- `component_health_processing`
- `primary_component`

This metadata makes it possible to study:

- behavior classification
- retrieval
- transition detection
- latent regression
- component reasoning
- clean vs transition robustness

## Current Data Scale

The current default `v4` dataset configuration uses:

- `60` trajectories
- `240` events per trajectory
- `14400` raw events total
- `1380` windows after windowing
- `20 x 6` event-feature windows after preprocessing

The dataset is generated by:

[01_generate_raw_logs_v4.py](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\scripts\01_generate_raw_logs_v4.py)

using:

[v4.yaml](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\configs\datasets\v4.yaml)

## Current Strengths

The strongest properties of the current `v4` design are:

- behavior is defined by process dynamics rather than a short explicit template
- transitions are real and meaningful
- multiple behavior classes share the same event vocabulary
- hidden component health allows genuine cross-component failure logic
- state and severity are no longer exact reflections of latent state

## Current Limitations

Even after weakening state, severity, and direct event-emission bias, `v4` still contains strong learnable structure.

This is visible from strong baseline performance, especially for:

- TF-IDF n-gram representations
- PCA
- summary-statistic baselines

This suggests that the current `v4` still exposes some structured shortcuts, most likely through:

- smooth latent trends that remain easy to summarize
- numeric channels that remain strongly informative
- stable recovery or spread geometries that shallow models can still exploit

That is not necessarily a flaw, but it means `v4` is best viewed as:

> a more realistic and process-grounded synthetic benchmark, not yet a fully shortcut-resistant one.

## Recommended Interpretation

`v4` should be understood as a benchmark that tests whether embeddings can capture:

- recovery speed
- delayed stabilization
- repeated instability
- cross-component spread
- ambiguous transition regions

rather than simply memorizing a short pattern template.

At the same time, `v4` remains controlled enough that we can diagnose where shallow baselines still succeed and where learned sequence models begin to add value.

## Files and Entry Points

Implementation:

- [generator_v4.py](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\src\behavior_log\generation\generator_v4.py)

Generation script:

- [01_generate_raw_logs_v4.py](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\scripts\01_generate_raw_logs_v4.py)

Configs:

- [datasets/v4.yaml](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\configs\datasets\v4.yaml)
- [preprocessing/v4.yaml](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\configs\preprocessing\v4.yaml)
- [windowing/v4.yaml](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\configs\windowing\v4.yaml)

Example generated artifacts:

- [raw_logs.csv](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\artifacts\datasets\latent_behavior_sequence_v4\raw\raw_logs.csv)
- [dataset_manifest.json](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\artifacts\datasets\latent_behavior_sequence_v4\raw\dataset_manifest.json)
- [windows.npz](E:\thesis project\log-to-vec-benchmark\experiments\behavior_log\artifacts\datasets\latent_behavior_sequence_v4\windows\windows.npz)
