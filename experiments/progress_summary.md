## 1. Overall Pipeline Summary

The current project pipeline is designed to convert behavior-driven event logs into fixed-dimensional window embeddings and evaluate whether these embeddings capture meaningful behavioral structure. The pipeline is implemented as a staged workflow:

1. Generate behavior-driven raw logs  
2. Preprocess each log event into a numeric feature vector  
3. Create fixed-length sliding log windows  
4. Split the data by `trajectory_id`  
5. Train or compute window-level representations  
6. Extract window embeddings  
7. Evaluate the embedding space  


## 2. Data Generation

The first stage of the pipeline generates behavior-driven synthetic raw event logs. The generated data are sequential event logs, not independent event samples. The generator creates multiple trajectories, and each trajectory contains several behavior cycles. Each cycle is an 8-event sequence with a specific behavior pattern.

The current generator is designed around long-range behavior outcomes. This means that a behavior is not determined by one single event, but by the order of events across the whole cycle. For example, two behaviors may start with similar warning or recovery events, but they differ in the final outcome, such as whether the system returns to `RUNNING` or remains in `WARNING`.

### 2.1 Generated data structure

Each row in the raw dataset represents one log event. The generated raw log contains the following columns:

| Column | Meaning |
|---|---|
| `timestamp` | Timestamp of the event. |
| `trajectory_id` | ID of the trajectory. Events with the same `trajectory_id` belong to the same ordered log sequence. |
| `cycle_id` | ID of the behavior cycle inside one trajectory. |
| `cycle_behavior_id` | Behavior label assigned to the whole cycle. |
| `event_type` | Type of log event, such as sensor reading, alarm, actuator command, or communication status. |
| `component` | System component related to the event. This is derived from `event_type`. |
| `severity` | Severity level of the event. |
| `state` | System state at the event position. |
| `message` | Text description generated from the event type and state. |
| `behavior_id` | Behavior label assigned to this event. |
| `sensor_value` | Numeric sensor-related value. |
| `control_value` | Numeric control-related value. |
| `is_transition` | Event-level transition flag. In the current generator, this is set to `False` for generated rows. |

### 2.2 Behavior design

The dataset contains four behavior classes. Each behavior represents a different system outcome pattern.

| Behavior ID | Behavior name | Main meaning |
|---|---|---|
| `0` | Successful Recovery | The system detects an alarm, enters recovery, and finally clears the alarm. The final state returns to `RUNNING`. |
| `1` | Failed Recovery | The system detects an alarm and attempts recovery, but the alarm appears again near the end. The final state remains `WARNING`. |
| `2` | Successful Communication Recovery | The system experiences communication errors, recovers communication, and finally returns to `RUNNING`. |
| `3` | Persistent Communication Failure | The system experiences communication errors and temporary communication recovery, but the final outcome is still a communication error. The final state remains `WARNING`. |

Each behavior has three variants. The variants keep the same high-level outcome but slightly change the order of intermediate events. This prevents the task from becoming a simple fixed-pattern matching problem.

### 2.3 Behavior patterns

Each behavior cycle contains 8 main events. The table below shows the behavior logic using representative variants.

| Behavior ID | Representative event pattern | State pattern | Outcome |
|---|---|---|---|
| `0` | `SENSOR_READ → ALARM_TRIGGER → ACTUATOR_CMD → SENSOR_READ → WATCHDOG_TICK → ALARM_TRIGGER → WATCHDOG_TICK → ALARM_CLEAR` | `RUNNING → WARNING → RECOVERY → RECOVERY → RECOVERY → RECOVERY → RECOVERY → RUNNING` | Alarm is cleared successfully. |
| `1` | `SENSOR_READ → ALARM_TRIGGER → ACTUATOR_CMD → SENSOR_READ → WATCHDOG_TICK → ALARM_CLEAR → WATCHDOG_TICK → ALARM_TRIGGER` | `RUNNING → WARNING → RECOVERY → RECOVERY → RECOVERY → RECOVERY → RECOVERY → WARNING` | Recovery fails because the alarm appears again. |
| `2` | `SENSOR_READ → COMMUNICATION_ERROR → WATCHDOG_TICK → SENSOR_READ → COMMUNICATION_OK → WATCHDOG_TICK → COMMUNICATION_ERROR → COMMUNICATION_OK` | `RUNNING → WARNING → WARNING → WARNING → RECOVERY → RECOVERY → RECOVERY → RUNNING` | Communication recovers successfully. |
| `3` | `SENSOR_READ → COMMUNICATION_ERROR → WATCHDOG_TICK → SENSOR_READ → COMMUNICATION_OK → WATCHDOG_TICK → COMMUNICATION_OK → COMMUNICATION_ERROR` | `RUNNING → WARNING → WARNING → WARNING → RECOVERY → RECOVERY → RECOVERY → WARNING` | Communication failure persists. |

The key difference is the long-range outcome. For example, behavior `0` and behavior `1` both contain alarm and recovery events, but behavior `0` ends with `ALARM_CLEAR` and returns to `RUNNING`, while behavior `1` ends with another `ALARM_TRIGGER` and remains in `WARNING`. Similarly, behavior `2` and behavior `3` both contain communication errors and communication recovery events, but behavior `2` ends with `COMMUNICATION_OK`, while behavior `3` ends with `COMMUNICATION_ERROR`.

### 2.4 Event types

The generator uses seven event types:

| Event type | Meaning |
|---|---|
| `SENSOR_READ` | A sensor unit samples the process. |
| `ALARM_TRIGGER` | A sensor unit raises an alarm. |
| `ALARM_CLEAR` | A sensor unit clears the alarm. |
| `ACTUATOR_CMD` | A control unit issues an actuator command. |
| `WATCHDOG_TICK` | A control unit emits a watchdog tick. |
| `COMMUNICATION_OK` | A network unit reports healthy communication. |
| `COMMUNICATION_ERROR` | A network unit reports a communication error. |

### 2.5 Component mapping

The `component` column is not sampled independently. It is determined by the `event_type`.

| Event type | Component |
|---|---|
| `SENSOR_READ` | `SensorUnit` |
| `ALARM_TRIGGER` | `SensorUnit` |
| `ALARM_CLEAR` | `SensorUnit` |
| `ACTUATOR_CMD` | `ControlUnit` |
| `WATCHDOG_TICK` | `ControlUnit` |
| `COMMUNICATION_OK` | `NetworkUnit` |
| `COMMUNICATION_ERROR` | `NetworkUnit` |

This means that event types related to sensing and alarms are mapped to `SensorUnit`, control-related events are mapped to `ControlUnit`, and communication events are mapped to `NetworkUnit`.

### 2.6 State design

The generator uses three system states:

| State | Meaning |
|---|---|
| `RUNNING` | The system is operating normally. |
| `WARNING` | The system is in a warning or error-related condition. |
| `RECOVERY` | The system is in a recovery process. |

The state is aligned with the event sequence. For example, alarm or communication error events usually appear around `WARNING`, while actuator commands and watchdog ticks often appear during `RECOVERY`.

### 2.7 Severity design

The `severity` column is generated from the event type.

| Event type group | Severity rule |
|---|---|
| `ALARM_TRIGGER`, `COMMUNICATION_ERROR` | Usually `WARNING`, with a small probability of `ERROR`. |
| Other event types | `INFO`. |

Specifically, `ALARM_TRIGGER` and `COMMUNICATION_ERROR` have an 8% probability of being assigned `ERROR`; otherwise they are assigned `WARNING`. All other event types are assigned `INFO`.

### 2.8 Numeric fields

The generator produces two numeric columns:

| Numeric field | Meaning |
|---|---|
| `sensor_value` | Sensor-related numeric value. |
| `control_value` | Control-related numeric value. |

These numeric values are event-dependent.

For `SENSOR_READ`, `sensor_value` follows a behavior-specific trend:

| Behavior ID | Sensor trend for `SENSOR_READ` |
|---|---|
| `0` | Strong decreasing trend during successful recovery. |
| `1` | Weaker decrease at first, then rises again near the end, reflecting failed recovery. |
| `2` | Moderate decreasing trend during successful communication recovery. |
| `3` | Slight decrease at first, then rises again near the end, reflecting persistent communication failure. |

For `ALARM_TRIGGER`, `sensor_value` is high, around 86 with random noise.

For `ACTUATOR_CMD`, `control_value` is generated around 45 with random noise.

For most other event types, the numeric value remains 0 unless specifically generated by the above rules.

In the raw CSV, these values are stored as two separate columns: `sensor_value` and `control_value`. In the preprocessing stage, they are placed after the encoded categorical fields. Therefore, each event is later represented as:

`[event_type_id, component_id, severity_id, state_id, sensor_value, control_value]`

### 2.9 Noise event design

The generator also inserts noise events into the sequence. Noise events are sampled from:

| Noise event type |
|---|
| `WATCHDOG_TICK` |
| `COMMUNICATION_OK` |
| `SENSOR_READ` |

These noise events are inserted into gaps between regular behavior-pattern events. They keep the same behavior label as the surrounding cycle, but they make the sequence less clean and less deterministic.

### 2.10 Current generation configuration

The current dataset is generated with the following configuration:

```yaml
seed: 42
num_trajectories: 60
events_per_trajectory: 200
behavior_ids: [0, 1, 2, 3]
behavior_probs: [0.25, 0.25, 0.25, 0.25]
trajectory_spacing_seconds: 600.0
min_step_seconds: 1.0
max_step_seconds: 3.0
noise_event_prob: 0.15

The meaning of each parameter is summarized below:

| Parameter | Meaning |
|---|---|
| `seed` | Random seed used to make the generated dataset reproducible. |
| `num_trajectories` | Number of independent trajectories to generate. |
| `events_per_trajectory` | Number of events in each trajectory. |
| `behavior_ids` | Set of behavior classes used in the dataset. |
| `behavior_probs` | Sampling probability for each behavior class. |
| `trajectory_spacing_seconds` | Time gap between the start times of different trajectories. |
| `min_step_seconds` | Minimum time interval between two consecutive events inside one trajectory. |
| `max_step_seconds` | Maximum time interval between two consecutive events inside one trajectory. |
| `noise_event_prob` | Probability of injecting a noisy event into the generated sequence. |

With this configuration, the generator creates `60` trajectories with `200` events in each trajectory. Therefore, the raw dataset contains approximately `12,000` events in total.

The four behavior classes are sampled with equal probability, so the dataset is balanced at the behavior level. Within each trajectory, the time interval between consecutive events is randomly sampled between `1` and `3` seconds. The noise probability is set to `0.15`, meaning that noisy events are injected into the generated sequences.

### 2.7 Output of this stage

The output of the generation stage includes:

1. A raw log CSV file.
2. A metadata file.
3. A dataset manifest for reproducibility.

This raw dataset is then passed to the preprocessing stage, where categorical log fields are encoded and numeric fields are normalized.





