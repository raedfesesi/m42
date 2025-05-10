✅ **Summary of the Anomaly Detection Results:**
- **Total records processed:** 240,796
- **Total anomalies detected:** 2,408 (about 1%) — this aligns with the `contamination=0.01` parameter we set in the `IsolationForest`, meaning the model was looking for the most extreme ~1% of data points.
- **Anomalies identified:** Rows with notably extreme `result_value`, `percentage`, or `median` metrics, compared to their group-level means and standard deviations.
- **Sample anomalous entries (like AD24-00116.026, AD23-00429.033):** These records had unusually high combinations of result counts, percentages, or median values, standing far apart from the typical distribution, which flagged them as outliers.

✅ **Why EDA mattered before anomalies:**
- We used EDA to understand the spread (mean, std) and the value ranges (min, max) of the key numerical columns.
- Knowing the high variability and extreme ranges in some groups (large `std` values) was crucial; without scaling (via `StandardScaler`), the algorithm might have biased toward large-magnitude fields.
- By identifying fields with sufficient variability and meaningful numeric distributions, we chose `result_value`, `percentage`, and `median` as the right input features for the anomaly model.

✅ **Next step (visual integration suggestion):**
- Add scatter or line plots over time, highlighting anomaly points (e.g., with red markers) on the timeline or by group.
- Visualize the distribution of anomaly scores across the dataset to spot how tightly or widely anomalies are spread.
