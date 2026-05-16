# Routing Locality Experiment

Date: 2026-05-15

## Setup

- Dataset: deterministic MNIST PCA-16 phase cache from `notebooks/repeated_encoding_tutorials/cache/mnist_pca16_phase_no_clip_rng7.npz`
- Model: 5-layer repeated phase optical classifier
- Width: 16
- Routing limit: 2, meaning output port `j` is allowed for input port `i` when `|j - i| <= 2`
- Training: 140 epochs, Adam, learning rate `5e-3`, batch size `1000`, seed `7`
- Result cache: `artifacts/routing_penalty_mnist_results_corrected.json`

## Penalties Compared

- `fractional`: mean fraction of transmitted power outside the allowed routing band
- `distance_weighted`: mean distance-weighted fraction of transmitted power outside the allowed routing band
- `absolute`: mean absolute optical power outside the allowed routing band

The `absolute` mode was included as a negative control for subunitary layers. It can reduce the penalty by increasing insertion loss, so it is not recommended as the default for passive lossy layers.

## Results

| Model | Penalty | Lambda | Val Acc | Val Loss | Fractional Leakage | Mean Insertion Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unitary repeated | none | 0.00 | 93.22% | 0.2177 | 70.27% | 0.000 dB |
| Unitary repeated | fractional | 0.20 | 93.09% | 0.2198 | 28.78% | 0.000 dB |
| Unitary repeated | distance-weighted | 0.05 | 93.18% | 0.2239 | 33.99% | 0.000 dB |
| Subunitary repeated | none | 0.00 | 93.77% | 0.2008 | 72.04% | 2.086 dB |
| Subunitary repeated | absolute | 0.20 | 93.43% | 0.2153 | 29.74% | 3.198 dB |
| Subunitary repeated | fractional | 0.20 | 92.97% | 0.2229 | 28.14% | 1.994 dB |
| Subunitary repeated | distance-weighted | 0.05 | 92.82% | 0.2221 | 33.29% | 1.825 dB |

## Conclusion

Use `fractional` routing penalty as the default.

It strongly reduces nonlocal routing leakage while avoiding the main failure mode of absolute leakage: extra subunitary insertion loss. In this controlled run, subunitary absolute leakage increased mean insertion loss from `2.086 dB` to `3.198 dB`, while fractional leakage slightly reduced insertion loss to `1.994 dB`.

Use `distance_weighted` only when the physical layout cost grows with routing distance. It is defensible, but in this run it produced slightly weaker locality and lower accuracy than plain fractional leakage.

## Local Mesh Trial

I also tested replacing each dense unitary with a physical nearest-neighbor Clements mesh. This does not add a routing penalty and does not explicitly zero matrix entries. Locality is controlled structurally by the mesh depth.

- Result cache: `artifacts/local_mesh_mnist_results.json`
- Same dataset, classifier depth, optimizer, batch size, and seed as above
- Mesh depth is the number of nearest-neighbor Clements layers inside each optical layer

| Model | Mesh Depth | Val Acc | Val Loss | Fractional Leakage | Mean Insertion Loss | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense unitary | n/a | 93.22% | 0.2177 | 70.27% | 0.000 dB | 5121 |
| Local Clements mesh | 2 | 77.00% | 0.7543 | 0.00% | 0.000 dB | 241 |
| Local Clements mesh | 4 | 87.30% | 0.3995 | 42.35% | 0.000 dB | 401 |
| Local Clements mesh | 8 | 92.18% | 0.2555 | 56.61% | 0.000 dB | 721 |
| Local Clements mesh | 16 | 93.95% | 0.1917 | 68.15% | 0.000 dB | 1361 |

The local mesh is physically clean and has no unintended insertion loss because it remains unitary. The tradeoff is expressivity versus locality: depth 2 satisfies the `routing_limit=2` locality target but loses too much accuracy, while depth 16 recovers accuracy but becomes almost as nonlocal as the dense unitary.

For this MNIST PCA-16 task, structural locality alone did not produce the desired `~10%` leakage and high accuracy at the same time. The best practical route is likely a hybrid: use a shallow-to-medium local mesh for physical bias, plus a fractional routing penalty or leakage target to keep deeper meshes local during training.

## High-Accuracy Locality Trials

I then ran trial-and-error targeted experiments instead of a grid sweep. The goal was:

- MNIST validation accuracy at or above `93%`
- Fractional routing leakage at or below `10%`
- For subunitary layers, no insertion-loss increase relative to the guarded baseline

The successful objective was a linear hinge on fractional leakage:

```text
L = L_task
  + lambda_route * max(leakage - 0.10, 0)
  + lambda_loss * max(mean_insertion_loss_db - 2.286 dB, 0)^2
```

The best trial used dense trainable subunitary layers, not a local mesh:

| Model | Optical Depth | Route Objective | Lambda Route | Val Acc | Val Loss | Fractional Leakage | Mean Insertion Loss |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Subunitary repeated | 8 | linear hinge | 2.3 | 93.31% | 0.2205 | 9.92% | 2.167 dB |

The full trial log is saved in `artifacts/high_accuracy_locality_trials.json`.

The local-mesh and unitary-target trials got close but did not meet both constraints simultaneously. The most useful local-mesh run compressed leakage to `10.72%`, but accuracy dropped to `91.69%`. The best unitary target run reached `92.94%` at `10.78%` leakage.

Conclusion: for this task, the cleanest high-accuracy local solution is currently a deeper subunitary repeated model with a fractional leakage target and insertion-loss guard. This is more effective than pure structural locality because it preserves enough optical degrees of freedom while directly enforcing the physical locality criterion.

## Five-Layer Constraint Check

I rejected the 8-layer direction and reran targeted trials with at most five optical layers. Under this stricter constraint, the experiments did not find a model that simultaneously reached `>=93%` validation accuracy and `<=10%` fractional routing leakage.

Representative 5-layer results:

| Model | Objective | Val Acc | Val Loss | Fractional Leakage | Mean Insertion Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Final-subunitary phase | none | 94.00% | 0.1892 | 69.93% | 0.338 dB |
| Final-subunitary phase | linear hinge, lambda 2.3 | 92.43% | 0.2454 | 10.01% | 0.192 dB |
| All-subunitary phase | none | 94.28% | 0.1856 | 71.59% | 2.494 dB |
| All-subunitary phase | linear hinge, lambda 2.3 | 92.43% | 0.2418 | 9.99% | 2.239 dB |
| All-subunitary phase | warmup plus low-LR compression | 92.74% | 0.2282 | 10.00% | 1.671 dB |
| All-subunitary phase | linear hinge, 600 epochs | 91.37% | 0.2576 | 9.96% | 2.252 dB |

The 5-layer models have enough capacity for high accuracy when routing is unconstrained, and enough capacity for `~10%` leakage when locality is enforced. The issue is the intersection: enforcing `~10%` per-layer leakage removes too much useful mixing for the 5-layer all-optical classifier.

Within the current model family, the best 5-layer compromise was the all-subunitary phase model with warmup and lower learning-rate compression:

```text
accuracy = 92.74%
leakage = 10.00%
mean insertion loss = 1.671 dB
```

This is close, but it does not satisfy the `>=93%` accuracy target. The result suggests that if five layers is a hard limit, the next viable changes should alter the representation or readout, not just tune the routing coefficient.

## Wider Locality Band: Routing Limit 4

I repeated the five-layer search with `routing_limit=4`. This means each input port can route to up to nine local output ports:

```text
i - 4 <= j <= i + 4
```

The best strict result used the all-subunitary phase model with a linear hinge routing objective:

| Model | Routing Limit | Lambda Route | Val Acc | Val Loss | Fractional Leakage | Mean Insertion Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-subunitary phase | 4 | 0.5 | 93.36% | 0.2092 | 9.90% | 2.275 dB |

Near-boundary trials:

| Model | Lambda Route | Val Acc | Fractional Leakage |
| --- | ---: | ---: | ---: |
| All-subunitary phase | 0.35 | 93.69% | 10.03% |
| All-subunitary phase | 0.40 | 93.71% | 10.11% |
| All-subunitary phase | 0.50 | 93.36% | 9.90% |

So for the five-layer constraint, widening the local band from `routing_limit=2` to `routing_limit=4` crosses the target: accuracy remains above `93%` while leakage drops below `10%`.

The full trial log is saved in `artifacts/high_accuracy_locality_limit4_trials.json`.

## Clean Lumix API

The routing-locality API should expose physical measurements, not experimental loss variants.

Public functional utilities:

```python
from lumix.functional.routing import routing_mask, routing_leakage
```

Public layer argument:

```python
routing_limit=4
```

Supported on:

```python
UnitaryLinear(...)
SubUnitaryLinear(...)
ClementsLinear(...)
```

When `routing_limit` is set, layers report:

```python
metrics["routing_leakage"]
```

Training code reads the Linen metrics collection through the stateless Lumix metrics adapter:

```python
from lumix.metrics import AboveTarget, MetricCollection

metrics = MetricCollection.from_linen(updates["metrics"])
routing_leakage = metrics["routing_leakage"].compute()

loss = task_loss
loss += lambda_route * AboveTarget(target_leakage)(routing_leakage)
loss += lambda_loss * AboveTarget(loss_guard_db)(mean_insertion_loss_db) ** 2
```

Equivalently, the objective is:

```text
L = L_task
  + lambda_route * max(routing_leakage - target_leakage, 0)
  + lambda_loss * max(mean_insertion_loss_db - loss_guard_db, 0)^2
```

The layers do not report `routing_penalty` and do not accept `routing_penalty_mode`. This keeps Lumix focused on optical modules and physical metrics while leaving optimization policy in the training workflow.
