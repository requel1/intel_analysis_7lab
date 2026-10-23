import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================ Загрузка ============================
COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]

NUM_COLS = ["age", "fnlwgt", "education_num", "capital_gain",
            "capital_loss", "hours_per_week"]


def load_census(path):
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True)
    if str(df.iloc[0]["age"]).strip().lower() == "age":
        df = df.iloc[1:].reset_index(drop=True)

    df["income"] = (df["income"].astype(str)
                    .str.replace(".", "", regex=False).str.strip())
    df = df.replace("?", np.nan)
    df = df.dropna().reset_index(drop=True)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    for c in df.columns:
        if c not in NUM_COLS and c != "income":
            df[c] = df[c].astype(str).str.strip()

    return df


# ============================ Кодирование ============================
def fit_encoders(df):
    """Обучает LabelEncoder на всех нечисловых колонках и на income."""
    encoders = {}
    for c in df.columns:
        if c == "income":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            le = LabelEncoder()
            le.fit(df[c].astype(str))
            encoders[c] = le
    target_le = LabelEncoder()
    target_le.fit(df["income"].astype(str))
    return encoders, target_le


def transform(df, encoders, target_le):
    """Применяет обученные энкодеры. Возвращает df с числовым income."""
    out = df.copy()
    for c, le in encoders.items():
        out[c] = le.transform(out[c].astype(str))
    out["income"] = target_le.transform(out["income"].astype(str))
    return out


# ============================ Критерии ============================
CRITERIA = {
    "id3":  "entropy",
    "c45":  "entropy",
    "cart": "gini",
}


def build_tree(X, y, criterion_key):
    clf = DecisionTreeClassifier(
        criterion=CRITERIA[criterion_key],
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def metrics(y_true, y_pred):
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_tree_png(clf, feature_names, class_names, title, path, max_depth=3):
    plt.figure(figsize=(22, 12))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, fontsize=8, max_depth=max_depth)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[OK] дерево сохранено: {path}")


# ============================ Эксперимент 1 ============================
def experiment_full(train_df, test_df, out_dir):
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ 1: обучение на 100 % данных, три критерия")
    print("=" * 80)

    encoders, target_le = fit_encoders(train_df)
    train_enc = transform(train_df, encoders, target_le)
    test_enc = transform(test_df, encoders, target_le)

    X_train = train_enc.drop(columns=["income"])
    y_train = train_enc["income"]
    X_test = test_enc.drop(columns=["income"])
    y_test = test_enc["income"]

    print(f"y_train dtype: {y_train.dtype}, уникальные: {sorted(y_train.unique())}")
    print(f"y_test  dtype: {y_test.dtype}, уникальные: {sorted(y_test.unique())}")

    rows = []
    for key in ["id3", "c45", "cart"]:
        t0 = time.time()
        clf = build_tree(X_train, y_train, key)
        elapsed = time.time() - t0
        pred = clf.predict(X_test)
        m = metrics(y_test, pred)
        m.update({
            "criterion": key,
            "sklearn_criterion": CRITERIA[key],
            "depth": int(clf.get_depth()),
            "leaves": int(clf.get_n_leaves()),
            "time_fit": round(elapsed, 3),
        })
        rows.append(m)
        print(f"[{key.upper():4}] acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} "
              f"depth={m['depth']} leaves={m['leaves']} time={elapsed:.2f}s")

        plot_tree_png(
            clf, list(X_train.columns), list(target_le.classes_),
            f"Дерево решений ({key.upper()}), обучение 100 %",
            os.path.join(out_dir, f"tree_{key}_full.png"),
            max_depth=3,
        )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "exp1_full.csv"), index=False)
    return rows


# ============================ Эксперимент 2 ============================
def experiment_splits(train_df, test_df, out_dir, criterion_key="c45"):
    print("\n" + "=" * 80)
    print(f"ЭКСПЕРИМЕНТ 2: доли 60/40 .. 90/10 (критерий = {criterion_key.upper()})")
    print("=" * 80)

    full = pd.concat([train_df, test_df], ignore_index=True)
    encoders, target_le = fit_encoders(full)
    full_enc = transform(full, encoders, target_le)

    feature_names = [c for c in full_enc.columns if c != "income"]
    X = full_enc[feature_names].values
    y = full_enc["income"].values

    rows = []
    for share in [0.6, 0.7, 0.8, 0.9]:
        n = len(X)
        n_train = int(n * share)
        rng = np.random.RandomState(42)
        idx = rng.permutation(n)
        tr, ts = idx[:n_train], idx[n_train:]

        clf = build_tree(X[tr], y[tr], criterion_key)
        pred = clf.predict(X[ts])
        m = metrics(y[ts], pred)
        m.update({
            "criterion": criterion_key,
            "train_share": share,
            "test_share": round(1 - share, 2),
            "n_train": n_train,
            "n_test": n - n_train,
            "depth": int(clf.get_depth()),
            "leaves": int(clf.get_n_leaves()),
        })
        rows.append(m)
        print(f"train={share*100:2.0f}%/test={(1-share)*100:2.0f}% "
              f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
              f"rec={m['recall']:.4f} f1={m['f1']:.4f} "
              f"depth={m['depth']} leaves={m['leaves']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"exp2_splits_{criterion_key}.csv"), index=False)
    plot_metrics_vs_split(df, out_dir, criterion_key)

    # дерево, обученное на 90 %
    clf90 = build_tree(X[tr], y[tr], criterion_key)
    plot_tree_png(clf90, feature_names, list(target_le.classes_),
                  f"Дерево решений ({criterion_key.upper()}), обучение 90 %",
                  os.path.join(out_dir, f"tree_{criterion_key}_90.png"),
                  max_depth=3)
    return df


def plot_metrics_vs_split(df, out_dir, criterion_key):
    x = (df["train_share"] * 100).astype(int)
    plt.figure(figsize=(10, 6))
    for col, label, marker in [
        ("accuracy", "Accuracy", "o"),
        ("precision", "Precision", "s"),
        ("recall", "Recall", "^"),
        ("f1", "F1", "D"),
    ]:
        plt.plot(x, df[col], marker=marker, linewidth=2, label=label)
    plt.xlabel("Доля обучающей выборки, %")
    plt.ylabel("Значение метрики")
    plt.title(f"Метрики качества от соотношения train/test ({criterion_key.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"metrics_vs_split_{criterion_key}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[OK] график сохранен: {path}")


# ============================ main ============================
def main():
    p = argparse.ArgumentParser(description="Дерево решений: ID3 / C4.5 / CART")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--criterion", choices=["id3", "c45", "cart", "all"], default="all")
    p.add_argument("--train-share", type=float, default=1.0)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Загрузка данных...")
    train_df = load_census(args.train)
    test_df = load_census(args.test)
    print(f"train: {len(train_df)} строк, test: {len(test_df)} строк")
    print(f"Классы (train): {train_df['income'].value_counts().to_dict()}")

    if args.criterion == "all" and args.train_share == 1.0:
        experiment_full(train_df, test_df, args.out)
    elif args.criterion == "all":
        experiment_splits(train_df, test_df, args.out, criterion_key="c45")
    else:
        if args.train_share == 1.0:
            encoders, target_le = fit_encoders(train_df)
            train_enc = transform(train_df, encoders, target_le)
            test_enc = transform(test_df, encoders, target_le)
            X_train = train_enc.drop(columns=["income"]); y_train = train_enc["income"]
            X_test = test_enc.drop(columns=["income"]); y_test = test_enc["income"]
            clf = build_tree(X_train, y_train, args.criterion)
            m = metrics(y_test, clf.predict(X_test))
            print(f"[{args.criterion.upper()}] {m}")
            plot_tree_png(clf, list(X_train.columns), list(target_le.classes_),
                          f"Дерево решений ({args.criterion.upper()}), обучение 100 %",
                          os.path.join(args.out, f"tree_{args.criterion}_full.png"))
        else:
            experiment_splits(train_df, test_df, args.out, criterion_key=args.criterion)


if __name__ == "__main__":
    main()