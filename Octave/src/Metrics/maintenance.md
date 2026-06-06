# Metric Maintenance

This file explains how to cleanly maintain the `Metrics/` folder.


## 1. File roles

The `Metrics/` folder is organized by responsibility.

```text
Metrics/
- base.py
- objective.py
- subjective.py
- MetricSet.py
- factory.py
- configs.py
- maintenance.md
```

### base.py

Shared parent class for all metrics.

Responsibilities:

- define common reductions. Current :
    - mean
    - sum
    - none (i.e keep the output as a non-scalar object)

- define common input checks. Current : 
    - shape compatibility
    - is the tensor single channel ?
    - are gradients tracked ?

- define common tensor-shape helpers.
    - squezze single channel to batched vector (i.e (B,1,T) -> (B,T))
    - batch-wise flattening (i.e (B1,B2,...,T) -> (B1 x B2 x ..., T))
    - tensor to numpy

Metric-specific logic should not be added here.

### objective.py

Objective / signal-level metrics.

Use this file for metrics based on waveform, spectral, or mathematical comparison. Prefer wrapping existing metric implementations rather than implementing custom ones.

Currently contains :
- SNR
- SI-SDR
- SI-SIR
- SI-SAR
- SD-SDR
- DTW
- $L_p$ Norms
- $L_p$ errors
- LSD
- Spectral KL divergence
- Itakura-Saito divergence


### subjective.py

Use this file for metrics based on perceptual scores or external evaluation backends.

Currently contains :
- STOI
- ESTOI
- PESQ
- DNSMOS P.808
- DNSMOS P.835 
    - SIG
    - BAK
    - OVRL


### MetricSet.py

Metric gathering and orchestration classes.

Responsibilities:

- MetricSet: stores and evaluates registered metrics;
- LoggableMetricSet: converts raw metric outputs into a flat logging dictionary;
- TSEMetricSet: provides the TSE interface using preds, target, mixture, and optionally clue.

This file should not instantiate metrics from config.

Lightning prefixes are handled outside Metrics/.


### loss.py

Loss aggregation logic.

Responsibilities:

- define loss objects built from metric values;
- aggregate selected metric values into a scalar training loss;
- store loss weights in a plain serializable form;
- enforce the minimization convention:
    - positive weight means the metric should be minimized;
    - negative weight means the metric should be maximized;
    - `None` or `0` weights mean the metric is inactive and should be ignored;
- return log-ready loss values, typically:
    - the total loss;
    - optionally each weighted loss term. (usefull for debugging)

A loss should not compute metrics directly.

A loss should consume the flat metric dictionary produced by `LoggableMetricSet`.

Lightning-specific prefixes should not be handled here.



### factory.py

Metric construction logic.

Responsibilities:

- map metric names to metric classes;
- instantiate metrics from config dictionaries;
- build metric sets from config dictionaries.

This file should contain all factory-related logic.

### configs.py

Default metric and loss configurations.

Responsibilities:

- store default metric and loss configs dictionaries;
- provide reusable config templates.

Experiment-specific changes should be done outside Metrics/, by copying and modifying these defaults.

### maintenance.md

Metric maintenance protocol.

Responsibilities:

- explain how to add metrics;
- explain how to delete metrics;
- document the minimal consistency checks required after changes.

## 2. Adding a new metric

Adding a metric should update only the files that are relevant to the metric pipeline.

### Step 1 : Choose the metric file

Add the metric class in:

- objective.py for objective / signal-level metrics;
- subjective.py for perceptual, external-library-based, or evaluation-only metrics.

### Step 2 : Implement the metric class

Every metric should inherit from BaseMetric.

The class should define:

- `__init__`
- forward
- check_inputs

The forward method should:

1. call self.check_inputs(...);
2. compute raw metric values;
3. return self.reduce(values) when the metric output is scalar-like.

### Step 3 : Handle dependencies cleanly

If the metric uses an optional or heavy dependency, import it inside `__init__`.

Store the imported function as an attribute.

Example:

```python
from some_package import some_function
self.some_function = some_function
```

### Step 4 : Add input checks

Use existing checks from BaseMetric when possible.

Evaluation-only metrics should call check_not_autograd_tracked.

If the new metric requires a check or helper that is likely to be reused by other metrics, add it to `base.py`.

Rule:

- reusable metric-level logic goes in `base.py`;
- metric-specific logic stays inside the metric class.

### Step 5 : Register the metric in factory.py

Import the new metric class in factory.py.

Then add it to the default metric registry with the format : "metric_name": MetricClass

The metric name should be lowercase and consistent with the rest of the project.

### Step 6 : Add a TSE input route if needed

If the metric should be usable through TSEMetricSet, add it to: TSEMetricSet.metric_to_input_names

### Step 7 : Add output names if needed

If the metric returns several outputs, add names in: TSEMetricSet.known_output_names

Only do this when the output is meant to be split for logging.

### Step 8 : Add it to config.py if needed

If the metric should be included by default, add it to the default metric config.

### Step 9 : Update dependencies if needed

If the metric requires a new package, add it to the appropriate requirements file.

Use requirements.in for regular Python dependencies.

Use the Torch-specific requirements file only for the PyTorch stack.

## 3. Deleting an old metric

Deleting a metric means removing it from every place where it is exposed to the metric pipeline.

### Step 1 : Remove it from config.py

If the metric appears in a default config, remove its entry.

### Step 2 : Remove it from factory.py

Remove the metric from the default registry.

Also remove its import if it is no longer used.

### Step 3 : Remove its TSE input route

If the metric appears in: TSEMetricSet.metric_to_input_names

remove its route.

### Step 4 : Remove its output names if needed

If the metric appears in: TSEMetricSet.known_output_names

remove its entry.

### Step 5 : Delete the metric class if unused

Delete the class from objective.py or subjective.py only if no other code still imports or uses it.

### Step 6 : Clean dependencies if needed

If the deleted metric was the only user of an external dependency, remove that dependency from the relevant requirements file.

### Step 7 : Check imports

After deletion, verify that no file still imports the removed metric.

Typical places to check:

- factory.py
- config.py
- notebooks
- experiment configs

## 4. Sanity check of the metric pipeline

??
TODO
??