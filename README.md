# 📦 Wholesale Customer Segmentation & Decision Engine

An end-to-end Machine Learning project transforming raw wholesale spending data into actionable business intelligence. This project identifies distinct customer segments using unsupervised learning (PCA & K-Means), engineers a supervised classification pipeline (Random Forest), and deploys an interactive decision dashboard via Streamlit to optimize delivery logistics and inventory management.

---

## 🚀 Live Demo & Repository
- **GitHub Repository:** [mahmoudnazmy48/wholesale-customer-segmentation](https://github.com/mahmoudnazmy48/wholesale-customer-segmentation)
- **Deployment:** Streamlit Cloud

---

## 📌 Project Overview & Objectives
A wholesale distributor collects annual spending across six product categories:
`Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, and `Delicatessen`.

The business challenges addressed in this project:
1. **Uncovering Latent Structure:** Identifying natural customer groupings without prior labels.
2. **Operational Efficiency:** Determining how service changes (e.g., delivery schedules) impact different client archetypes via targeted A/B testing design.
3. **Automated Onboarding:** Classifying newly acquired clients instantly without re-clustering the entire historical dataset.

---

## 🔬 Methodology & Workflow

### 1. Exploratory Data Analysis & Feature Scaling
- Inspected spending distributions and observed strong positive skewness across all categories.
- Applied Natural Logarithmic transformation (`np.log`) to normalize feature variances and stabilize outliers.
- Detected and handled multi-variable outliers using Tukey’s fence method (1.5 IQR).

### 2. Dimensionality Reduction (PCA)
- Reduced the 6-dimensional feature space into 2 principal dimensions:
  - **Dimension 1:** Heavily weighted towards `Grocery`, `Milk`, and `Detergents_Paper`.
  - **Dimension 2:** Heavily weighted towards `Fresh` and `Frozen` spending.
- Captured the majority of total data variance while eliminating multi-collinearity.

### 3. Clustering (K-Means)
- Evaluated cluster quality using Silhouette Analysis across various cluster counts ($k \in [2, 6]$).
- Identified $k = 2$ as the optimal separation point, revealing two distinct core clusters:
  - **Cluster 0 — Retailers & Supermarkets:** Characterized by high spending on packaged goods (`Grocery`), dairy (`Milk`), and cleaning supplies (`Detergents_Paper`).
  - **Cluster 1 — Hospitality & Food Service:** Characterized by dominant expenditure on perishable goods (`Fresh`) and frozen food (`Frozen`).

### 4. Transition from Unsupervised to Supervised Learning
- Engineered the derived cluster assignments as ground-truth target labels ($y \in \{0, 1\}$).
- Trained a **Random Forest Classifier** on log-transformed spending features to instantly categorize incoming customers.
- Achieved high cross-validated precision and recall, allowing rapid classification without recalculating PCA transforms on historical data.

---

## 💡 Business Impact & Logistics Optimization

| Metric / Attribute | Cluster 0: Retail / Supermarket | Cluster 1: Hospitality / Restaurant |
| :--- | :--- | :--- |
| **Primary Purchasing Focus** | Grocery, Milk, Detergents & Paper | Fresh Produce, Frozen Goods |
| **Inventory Shelf Life** | Long / Stable (Non-perishables) | Short / Highly Perishable |
| **Recommended Delivery Schedule** | **3 Days / Week** | **5 Days / Week** |
| **Cost & Efficiency Impact** | Reduces fleet miles and logistics costs | Ensures product freshness and zero stockouts |

### A/B Testing Strategy
Instead of testing a reduced 3-day delivery schedule randomly across the entire customer base, the experiment is stratified by cluster:
- **Treatment Group B (3 days/week):** Successfully rolled out to Cluster 0 (retailers) with minimal disruption.
- **Control Group A (5 days/week):** Retained for Cluster 1 (restaurants/cafes) to safeguard customer satisfaction and prevent churn.

---

## 🛠️ Project Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── train_and_save.py           # ML pipeline training and model serialization script
├── wholesale_model.joblib      # Serialized PCA transformer, classifier, and metadata bundle
├── requirements.txt            # Python dependencies for local setup & cloud deployment
└── README.md                   # Project documentation