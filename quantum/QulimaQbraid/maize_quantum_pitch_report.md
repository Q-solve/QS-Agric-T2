# Hybrid Quantum–Classical Maize Yield Forecasting

## Results and quantum-advantage assessment

### Executive summary

We evaluated a hybrid forecasting architecture for Kenyan maize yield. A classical rainfall model first predicts yield from rainfall and weather indicators. A quantum-kernel model then learns the remaining contextual residual using four small features. Final predictions are reconstructed as:

```text
final yield = rainfall baseline + quantum residual prediction
```

**Bottom line:** this experiment demonstrates a working quantum-kernel residual workflow, but it does **not** demonstrate quantum advantage. The quantum residual model modestly improves the rainfall-only baseline, yet it is outperformed by a classical contextual Random Forest. The result is best presented as a credible feasibility and benchmarking study—not as evidence that quantum computing currently improves maize-yield forecasting.

### Data and modeling scope

- Source: `qulima_maize_yield.csv`
- 1,995 observations
- 47 Kenyan counties in the supplied file
- Years 1981–2024
- Annual and `Seasonal aggregate` records
- Target: `yield_t_ha`
- The dataset does not explicitly label a March–June season. The notebook uses seasonal aggregate rows when present and otherwise falls back to annual county-year records.
- The notebook writes a clean table to `county_year_modeling_table.csv`.

The feature groups were separated as follows:

**Rainfall/weather features:** rainfall availability, January–June rainfall total and long-term average, rainfall anomaly, dekadal rainfall mean and standard deviation, coefficient of variation, wet/heavy/dry dekad counts, longest dry spell, and rainfall onset day.

**Contextual features:** county latitude, county longitude, maize area, and a deterministic county identity encoding for the quantum experiment.

Production volume, source URLs, and other post-harvest or administrative fields were excluded from modeling to reduce target leakage.

### Validation design

The data were ordered by year. The latest 20% of years were held out as a chronological test set. Imputation, scaling, and model fitting were performed using training data only. The rainfall baseline was fit first; training residuals were then computed as:

```python
training residual = observed yield - rainfall baseline prediction
```

The quantum model was trained only on those training residuals and evaluated on the held-out years. This is substantially safer than random row splitting for a time-dependent agricultural problem, although a stronger production study should use rolling-origin validation across multiple forecast years.

### Models evaluated

1. **Rainfall-only Ridge:** classical baseline using weather variables only.
2. **Contextual Random Forest:** classical contextual comparator using county coordinates and maize area.
3. **All-feature Ridge:** classical model using rainfall plus contextual features.
4. **Quantum residual model:** rainfall Ridge baseline plus a quantum-kernel residual correction.

The quantum model used four contextual inputs:

- `county_lat`
- `county_lon`
- `maize_area_ha`
- county identity encoding

They were imputed and scaled to `[0, π]`, encoded with a four-qubit, two-repetition linear-entanglement `ZZFeatureMap`, and compared with a Qiskit `FidelityQuantumKernel`. Kernel Ridge regression was applied to the resulting Gram matrices.

### Held-out results

| Model | RMSE | MAE | R² |
| --- | ---: | ---: | ---: |
| Classical contextual Random Forest | **0.7297** | **0.6053** | **0.2720** |
| Quantum residual model | 0.8143 | **0.6185** | 0.0935 |
| Rainfall-only Ridge | 0.8235 | 0.6188 | 0.0730 |
| All-feature Ridge | 0.8904 | 0.6899 | -0.0838 |

### Interpretation

Relative to the rainfall-only baseline, the quantum residual model achieved:

- RMSE reduction: approximately **1.1%**
- MAE reduction: approximately **0.05%**
- R² improvement: from **0.0730 to 0.0935**

This is a positive but small improvement. It is not enough to establish a robust performance gain because:

- only one chronological holdout was used;
- no repeated rolling-origin confidence intervals were computed;
- the quantum model was not compared against a fully tuned classical kernel method;
- the contextual Random Forest was materially better than the quantum model;
- the quantum kernel was evaluated with a local simulator, not quantum hardware;
- simulator kernel evaluation does not provide a computational speed advantage over classical kernels.

### Does this show quantum advantage?

**No—not in this experiment.** Quantum advantage requires more than a quantum circuit appearing in the pipeline. Depending on the claim, it could mean:

1. **Predictive advantage:** statistically reliable improvement over strong classical models on unseen data.
2. **Computational advantage:** lower runtime or better scaling at comparable accuracy.
3. **Economic/operational advantage:** better forecast quality per unit of compute, energy, or cost.

The current results satisfy none of these standards. The quantum model is better than the rainfall-only baseline by a narrow margin, but it trails the classical Random Forest. It also uses a simulator, where the fidelity kernel is computed classically and incurs substantial matrix-evaluation cost. Therefore, the defensible claim is:

> A small quantum-kernel residual model is technically feasible and shows a preliminary signal beyond a rainfall-only baseline, but no quantum advantage has been demonstrated against competitive classical alternatives.

### Why the quantum model may not win here

- The dataset is tabular, relatively small, and low-dimensional—conditions where tree ensembles and regularized classical models are highly competitive.
- County coordinates and maize area may have smooth or nonlinear relationships that Random Forest captures directly.
- The quantum model uses only four contextual variables and does not encode the complete rainfall feature space.
- Kernel Ridge hyperparameters were not extensively optimized.
- Quantum kernel matrices can become poorly conditioned or overly similar as feature dimension and circuit depth change.
- Historical yield contains socio-economic, agronomic, policy, technology, and measurement effects not represented in the feature set.
- A single test window makes it difficult to distinguish a real effect from time-period variation.

### What would be needed for a stronger quantum-advantage claim

1. Use multiple rolling-origin time splits, for example 2005–2015, 2006–2016, and so on, with mean performance and confidence intervals.
2. Tune every model under the same validation protocol, including Random Forest, Gradient Boosting, RBF-SVM, classical kernel Ridge, and Gaussian Processes.
3. Compare quantum kernels directly against classical RBF and polynomial kernels with matched feature inputs.
4. Perform ablations: rainfall-only, contextual-only, rainfall-plus-context, quantum residual, and quantum full-feature models.
5. Report kernel construction time, prediction time, memory use, and—if hardware is used—queue time, shot count, noise, and cost.
6. Evaluate multiple feature-map depths, entanglement structures, and regularization values without leaking test information.
7. Test on a locked future-year holdout or an external county-year dataset.
8. Quantify uncertainty and practical impact, such as identifying counties where forecast error changes planting, insurance, or food-security decisions.

### Pitch-safe conclusion

This project should be pitched as an **early hybrid quantum machine-learning feasibility study for climate-informed agricultural forecasting**. It shows how a physics-inspired quantum kernel can be inserted after a domain-specific rainfall baseline to model residual structure. The preliminary result is encouraging relative to the weakest baseline, but the classical contextual model remains the strongest performer.

The appropriate next-step message is:

> “We have established a reproducible hybrid architecture and benchmarked it honestly. The quantum residual model shows a small preliminary gain over a rainfall-only baseline, while classical nonlinear methods remain stronger. The next phase is a statistically powered rolling-validation and hardware-aware study to determine whether any quantum benefit survives against tuned classical kernels.”

### Reproducibility artifacts

- Executed notebook: `/home/jovyan/maize-quantum-forecasting-executed.ipynb`
- Metrics: `/home/jovyan/maize_quantum_outputs/model_metrics.csv`
- Predictions: `/home/jovyan/maize_quantum_outputs/test_predictions.csv`
- Modeling table: `/home/jovyan/maize_quantum_outputs/county_year_modeling_table.csv`
- Plot: `/home/jovyan/maize_quantum_outputs/model_comparison.png`
