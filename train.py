"""Smart Crop Recommendation System - training pipeline."""
import json, os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report)

os.makedirs("model", exist_ok=True)
os.makedirs("figs", exist_ok=True)

SEED = 42
FEATURES = ["N", "P", "K", "temperature", "ph", "rainfall"]      # the six inputs of the system
log = {}
# 1. Load and inspect
df = pd.read_csv("data/Crop_recommendation.csv")
log["shape"] = df.shape; log["missing"] = int(df.isna().sum().sum()); log["duplicates"] = int(df.duplicated().sum())
log["classes"] = sorted(df["label"].unique()); log["class_counts"] = df["label"].value_counts().to_dict()
log["describe"] = df.describe().round(2).to_dict()
# 2. Features / target and stratified split
X, y = df[FEATURES], df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
log["split"] = [len(X_tr), len(X_te)]
# 3. Models
models = {"Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
          "Gaussian Naive Bayes": GaussianNB(),
          "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED); res = {}; preds = {}
for name, m in models.items():
    m.fit(X_tr, y_tr); p = m.predict(X_te); preds[name] = p
    pr, rc, f1, _ = precision_recall_fscore_support(y_te, p, average="macro", zero_division=0)
    cv = cross_val_score(m, X_tr, y_tr, cv=skf, scoring="accuracy")
    res[name] = dict(acc=accuracy_score(y_te, p), prec=pr, rec=rc, f1=f1, train=accuracy_score(y_tr, m.predict(X_tr)), cv=cv.mean(), cvs=cv.std(), wrong=int((p != y_te).sum()))
log["results"] = res
# 4. Comparison with humidity included (all 7 columns of the dataset)
F7 = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X7tr, X7te = df.loc[X_tr.index, F7], df.loc[X_te.index, F7]; r7 = {}
for name in ["Gaussian Naive Bayes", "Random Forest"]:
    m = GaussianNB() if "Naive" in name else RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    m.fit(X7tr, y_tr); r7[name] = accuracy_score(y_te, m.predict(X7te))
log["with_humidity"] = r7
# 5. Details of the selected model
best = max(["Gaussian Naive Bayes", "Random Forest"], key=lambda k: res[k]["cv"]); log["best"] = best   # select by cross-validated accuracy
rf, nb = models["Random Forest"], models["Gaussian Naive Bayes"]
labels = list(rf.classes_)
for name in ["Random Forest", "Gaussian Naive Bayes"]:
    cm = confusion_matrix(y_te, preds[name], labels=labels)
    conf = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    log["confusions_" + ("rf" if "Forest" in name else "nb")] = sorted(conf, key=lambda t: -t[2])
log["report_rf"] = classification_report(y_te, preds["Random Forest"], output_dict=True, zero_division=0)
log["report_nb"] = classification_report(y_te, preds["Gaussian Naive Bayes"], output_dict=True, zero_division=0)
log["importance"] = dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)), key=lambda t: -t[1]))
log["class_means"] = df.groupby("label")[FEATURES].mean().round(1).to_dict(orient="index")
pickle.dump({"model": models[best], "features": FEATURES, "classes": labels}, open("model/crop_model.pkl", "wb"))
json.dump(log, open("results.json", "w"), indent=1, default=float)
# ---------------- figures
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 3.6)); c = df["label"].value_counts().sort_index(); plt.bar(c.index, c.values, color="#2d6a3f"); plt.xticks(rotation=60, ha="right", fontsize=8); plt.ylabel("Records"); plt.title("Class Distribution – 22 Crops, 100 Records Each"); plt.tight_layout(); plt.savefig("figs/classes.png", dpi=200); plt.close()
fig, ax = plt.subplots(2, 3, figsize=(10, 5.4))
for a, f in zip(ax.ravel(), FEATURES): sns.histplot(df[f], bins=30, kde=True, ax=a, color="#2f6db5"); a.set_title(f"Distribution of {f}", fontsize=10); a.set_xlabel("")
plt.tight_layout(); plt.savefig("figs/dist.png", dpi=200); plt.close()
sel = ["rice", "maize", "cotton", "jute", "coffee", "chickpea", "banana", "apple"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=df[df.label.isin(sel)], x="label", y="rainfall", order=sel, ax=ax[0], color="#9bb7d9"); ax[0].set_title("Rainfall (mm) by Crop"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=40)
sns.boxplot(data=df[df.label.isin(sel)], x="label", y="K", order=sel, ax=ax[1], color="#a8d5b5"); ax[1].set_title("Potassium (K) by Crop"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=40)
plt.tight_layout(); plt.savefig("figs/bycrop.png", dpi=200); plt.close()
plt.figure(figsize=(6, 4.8)); sns.heatmap(df[F7].corr(), annot=True, fmt=".2f", cmap="Blues", cbar=False, annot_kws={"size": 9}); plt.title("Correlation Heat Map of the Numerical Features"); plt.tight_layout(); plt.savefig("figs/corr.png", dpi=200); plt.close()
plt.figure(figsize=(9.6, 8.6)); sns.heatmap(confusion_matrix(y_te, preds["Random Forest"], labels=labels), annot=True, fmt="d", cmap="Greens", cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})
plt.xticks(rotation=60, ha="right", fontsize=8); plt.yticks(fontsize=8); plt.xlabel("Predicted crop"); plt.ylabel("Actual crop"); plt.title("Confusion Matrix – Random Forest (440 test records)"); plt.tight_layout(); plt.savefig("figs/cm_rf.png", dpi=200); plt.close()
plt.figure(figsize=(9.6, 8.6)); sns.heatmap(confusion_matrix(y_te, preds["Gaussian Naive Bayes"], labels=labels), annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})
plt.xticks(rotation=60, ha="right", fontsize=8); plt.yticks(fontsize=8); plt.xlabel("Predicted crop"); plt.ylabel("Actual crop"); plt.title("Confusion Matrix – Gaussian Naive Bayes (440 test records)"); plt.tight_layout(); plt.savefig("figs/cm_nb.png", dpi=200); plt.close()
names = ["Gaussian Naive Bayes", "Random Forest"]; mets = ["acc", "prec", "rec", "f1"]; xx = np.arange(4)
plt.figure(figsize=(7.4, 3.6))
for i, (n, col) in enumerate(zip(names, ["#2f6db5", "#2d6a3f"])):
    v = [res[n][k] * 100 for k in mets]; b = plt.bar(xx + (i - 0.5) * 0.36, v, 0.36, label=n, color=col)
    for bb, val in zip(b, v): plt.text(bb.get_x() + bb.get_width() / 2, val + 0.25, f"{val:.1f}", ha="center", fontsize=8)
plt.xticks(xx, ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]); plt.ylim(80, 102); plt.ylabel("Score on test set (%)"); plt.legend(loc="lower right"); plt.title("Model Comparison on the Held-out Test Set"); plt.tight_layout(); plt.savefig("figs/compare.png", dpi=200); plt.close()
imp = log["importance"]; k = list(imp)[::-1]; plt.figure(figsize=(6.4, 3.2)); plt.barh(k, [imp[i] for i in k], color="#2d6a3f")
for i, kk in enumerate(k): plt.text(imp[kk] + 0.004, i, f"{imp[kk]:.3f}", va="center", fontsize=9)
plt.xlim(0, max(imp.values()) * 1.18); plt.title("Random Forest Feature Importance"); plt.tight_layout(); plt.savefig("figs/importance.png", dpi=200); plt.close()
print(json.dumps({k: {m: round(v, 4) for m, v in r.items()} for k, r in res.items()}, indent=0)); print("with humidity", r7, "best", best, log["split"])
print("RF confusions", log["confusions_rf"]); print("NB confusions", log["confusions_nb"]); print("importance", {k: round(v, 3) for k, v in imp.items()})
