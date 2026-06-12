"""
Customer Segmentation Project
Thiranex - Skill Development & Future Tech
Due: 15 Jun 2026

Features:
- K-Means clustering on synthetic customer data
- Purchase pattern & demographic analysis
- Segment visualization (2D scatter + cluster profiles)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. GENERATE SYNTHETIC CUSTOMER DATA
# ─────────────────────────────────────────
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'age':              np.concatenate([np.random.normal(25, 3, 80),
                                        np.random.normal(45, 5, 100),
                                        np.random.normal(35, 4, 70),
                                        np.random.normal(60, 6, 50)]),
    'annual_income':    np.concatenate([np.random.normal(30000, 5000, 80),
                                        np.random.normal(75000, 10000, 100),
                                        np.random.normal(55000, 8000, 70),
                                        np.random.normal(90000, 12000, 50)]),
    'spending_score':   np.concatenate([np.random.normal(70, 10, 80),
                                        np.random.normal(40, 8, 100),
                                        np.random.normal(60, 12, 70),
                                        np.random.normal(20, 5, 50)]),
    'purchase_freq':    np.concatenate([np.random.normal(12, 2, 80),
                                        np.random.normal(6, 1.5, 100),
                                        np.random.normal(9, 2, 70),
                                        np.random.normal(3, 1, 50)]),
    'avg_order_value':  np.concatenate([np.random.normal(50, 10, 80),
                                        np.random.normal(150, 20, 100),
                                        np.random.normal(90, 15, 70),
                                        np.random.normal(200, 30, 50)]),
})

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# ─────────────────────────────────────────
# 3. FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ─────────────────────────────────────────
inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = K_range[np.argmax(silhouettes)]
print(f"✓ Optimal clusters: {best_k}  (silhouette = {max(silhouettes):.3f})")

# ─────────────────────────────────────────
# 4. FINAL K-MEANS MODEL
# ─────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data['segment'] = kmeans.fit_predict(X_scaled)

# Segment labels (business-friendly names)
seg_names = {0: 'Young Impulsives', 1: 'Wealthy Moderates',
             2: 'Mid-Range Actives', 3: 'Premium Seniors'}
data['segment_name'] = data['segment'].map(seg_names)

# ─────────────────────────────────────────
# 5. SEGMENT PROFILES
# ─────────────────────────────────────────
profile = data.groupby('segment_name').agg(
    Count=('age', 'count'),
    Avg_Age=('age', 'mean'),
    Avg_Income=('annual_income', 'mean'),
    Avg_Spending_Score=('spending_score', 'mean'),
    Avg_Purchase_Freq=('purchase_freq', 'mean'),
    Avg_Order_Value=('avg_order_value', 'mean'),
).round(1)
print("\n── Segment Profiles ──────────────────────")
print(profile.to_string())

# ─────────────────────────────────────────
# 6. VISUALIZATIONS  (4-panel figure)
# ─────────────────────────────────────────
COLORS = ['#4361EE', '#F72585', '#4CC9F0', '#7209B7']
palette = {name: COLORS[i % len(COLORS)]
           for i, name in enumerate(data['segment_name'].unique())}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segmentation Analysis', fontsize=18, fontweight='bold', y=0.98)
fig.patch.set_facecolor('#F8F9FA')
for ax in axes.flat:
    ax.set_facecolor('#FFFFFF')

# — Panel A: Elbow + Silhouette
ax1 = axes[0, 0]
color1 = '#4361EE'
ax1.plot(list(K_range), inertias, 'o-', color=color1, lw=2, label='Inertia')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1b = ax1.twinx()
ax1b.plot(list(K_range), silhouettes, 's--', color='#F72585', lw=2, label='Silhouette')
ax1b.set_ylabel('Silhouette Score', color='#F72585')
ax1b.tick_params(axis='y', labelcolor='#F72585')
ax1.axvline(best_k, color='#4CC9F0', linestyle=':', lw=2, label=f'Best k={best_k}')
ax1.set_title('A  Elbow & Silhouette Method', fontweight='bold')
ax1.set_xticks(list(K_range))

# — Panel B: Income vs Spending Score scatter
ax2 = axes[0, 1]
for seg_name, grp in data.groupby('segment_name'):
    ax2.scatter(grp['annual_income'], grp['spending_score'],
                c=palette[seg_name], label=seg_name, alpha=0.7, edgecolors='white', lw=0.4, s=55)
ax2.set_xlabel('Annual Income (₹)')
ax2.set_ylabel('Spending Score')
ax2.set_title('B  Income vs Spending Score', fontweight='bold')
ax2.legend(fontsize=8, framealpha=0.8)

# — Panel C: PCA 2-D projection
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
ax3 = axes[1, 0]
for seg_name, grp in data.groupby('segment_name'):
    idx = grp.index
    ax3.scatter(pcs[idx, 0], pcs[idx, 1],
                c=palette[seg_name], label=seg_name, alpha=0.75, s=50, edgecolors='white', lw=0.3)
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax3.set_title('C  PCA Cluster Projection', fontweight='bold')
ax3.legend(fontsize=8, framealpha=0.8)

# — Panel D: Segment size bar chart
ax4 = axes[1, 1]
counts = data['segment_name'].value_counts()
bars = ax4.barh(counts.index, counts.values,
                color=[palette[n] for n in counts.index], edgecolor='white', height=0.6)
for bar, val in zip(bars, counts.values):
    ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
             f'{val} customers', va='center', fontsize=9)
ax4.set_xlabel('Number of Customers')
ax4.set_title('D  Segment Size', fontweight='bold')
ax4.set_xlim(0, counts.max() + 30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = '/mnt/user-data/outputs/customer_segmentation_report.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Chart saved → {out_path}")

# ─────────────────────────────────────────
# 7. SAVE RESULTS CSV
# ─────────────────────────────────────────
csv_path = '/mnt/user-data/outputs/customer_segments.csv'
data.to_csv(csv_path, index=False)
print(f"✓ Data saved  → {csv_path}")
print("\n── Done! ──────────────────────────────────")
