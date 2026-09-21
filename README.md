# Downlink Throughput Prediction — 2nd Place, ITU AI/ML in 5G Challenge 2023

A PyTorch neural network that predicts cellular **downlink data rate** from radio measurements, network configuration, GPS and environmental context. It was built for the United Nations ITU AI/ML in 5G Challenge ("QoS Prediction," curated by Fraunhofer HHI) and **placed 2nd worldwide**.

| | |
|---|---|
| **Result** | 2nd place globally · R² ≈ 0.95 on held-out data |
| **Team** | Represented Shabodi (Summer 2023 internship) |
| **My role** | Built the modelling pipeline and ran the feature experiments; wrote the evaluation report and presented the findings to the ITU |
| **Stack** | Python · PyTorch · pandas · NumPy · scikit-learn · Matplotlib/Seaborn |

---

## The problem: https://challenge.aiforgood.itu.int/match/matchitem/80
Mobile networks can adapt ahead of time (to video bitrate, handovers, or vehicle-to-everything services) only if they can *predict* the quality of service a user is about to get. The challenge used the **Berlin V2X** dataset: drive-test measurements from vehicles on two commercial operators' networks in Berlin. Each record includes primary- and secondary-cell radio metrics, GPS position, weather and traffic. The task was to predict the downlink throughput actually achieved, including in environments the model hasn't seen.

## Approach

**1. Exploratory analysis.** Correlation heatmaps against uplink and downlink rates showed that the physical-layer signal metrics (RSRP, RSSI, SNR) were strongly related to throughput. RSRQ was the weakest of the four. Raw GPS features showed almost no direct correlation.

**2. Baseline.** A linear regression on primary-cell signal metrics fit poorly. That suggested the relationship is non-linear, so I moved to a neural network.

**3. Model.** A feed-forward network with one 64-unit hidden layer and ReLU, trained with Adam on MSE loss. Two changes cut error substantially: scaling the target from bits/s to Mbps, and training on both operators together instead of separately.

**4. Feature experiments.** I added feature groups one at a time and kept only those that improved held-out performance:

| Step | Features added | Outcome |
|---|---|---|
| A | Primary-cell RSRP, RSRQ, RSSI, SNR | Starting point |
| B | + Tx power, transport block size | Worse. Dropped |
| C | + **Secondary-cell** signal metrics, downlink MCS, ping | **R² = 0.945**. Biggest single gain |
| D | + Cell frequency and bandwidth | Small gain |
| E | + Altitude, humidity, cloud cover, traffic density | **R² ≈ 0.95** (24 features) |

The key insight was adding the secondary cell. Modern cells use carrier aggregation, so a device's throughput depends on *both* the primary and secondary cell. Adding the secondary cell's signal quality gave the largest improvement of any feature group.

**5. Generalization to New Environments.** A random train/test split is optimistic for drive-test data, because neighbouring samples are nearly identical. For a harder test, I trained on one area type and predicted another, i.e., park → avenue. This creates a more realistic measure of how the model would perform in a place it hasn't seen.

📄 **[Full report and slides](ITU_ML_Report.pdf)**



## What I'd do differently now
- **Spatially grouped validation** everywhere, not just in the cross-area test, to avoid leakage between neighbouring samples
- **Stronger tabular baselines** (XGBoost/LightGBM), which often match or beat MLPs on data like this
- **Feature scaling and hyperparameter search.** The network used raw feature scales and a fixed architecture
- **Feature importance** (SHAP or permutation) to replace one-at-a-time ablations
- **Temporal context**: throughput is autocorrelated, so recent history is a strong predictor
