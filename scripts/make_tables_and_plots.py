from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

MODEL_LOGS = {
    "Dropout 0.0": "models/model_0.pt",
    "Dropout 0.1": "models/model_0.1.pt",
    "Dropout 0.3": "models/model_0.3.pt",
    "Dropout 0.6": "models/model_0.6.pt",
    "Dropout 0.9": "models/model_0.9.pt",
}


def load_epoch_log(base_path: str, kind: str, label: str) -> pd.DataFrame:
    """
    kind should be 'train' or 'valid'
    Reads files like:
      models/model_0.pt.train.log
      models/model_0.pt.valid.log
    """
    path = f"{base_path}.{kind}.log"
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Epoch", "Loss", label],
        encoding="latin1",
        engine="python",
    )
    return df[["Epoch", label]]


def build_epoch_table(kind: str) -> pd.DataFrame:
    dfs = []

    for label, base in MODEL_LOGS.items():
        df = load_epoch_log(base, kind, label)
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Epoch", how="outer")

    merged = merged.sort_values("Epoch").reset_index(drop=True)
    return merged


def build_test_table() -> pd.DataFrame:
    row = {"Metric": "Test perplexity"}

    for label, base in MODEL_LOGS.items():
        path = f"{base}.test.log"
        df = pd.read_csv(
            path,
            sep="\t",
            encoding="latin1",
            engine="python",
        )
        row[label] = df.loc[0, "test_ppl"]

    return pd.DataFrame([row])


def plot_table(df: pd.DataFrame, title: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(10, 6))

    for col in df.columns:
        if col == "Epoch":
            continue
        plt.plot(df["Epoch"], df[col], label=col)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main() -> None:
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)

    train_table = build_epoch_table("train")
    valid_table = build_epoch_table("valid")
    test_table = build_test_table()

    train_table.to_csv(out_dir / "train_perplexity_table.csv", index=False)
    valid_table.to_csv(out_dir / "valid_perplexity_table.csv", index=False)
    test_table.to_csv(out_dir / "test_perplexity_table.csv", index=False)

    (out_dir / "train_perplexity_table.md").write_text(
        train_table.to_markdown(index=False),
        encoding="utf-8",
    )
    (out_dir / "valid_perplexity_table.md").write_text(
        valid_table.to_markdown(index=False),
        encoding="utf-8",
    )
    (out_dir / "test_perplexity_table.md").write_text(
        test_table.to_markdown(index=False),
        encoding="utf-8",
    )

    plot_table(
        train_table,
        title="Training Perplexity over Epochs",
        ylabel="Training Perplexity",
        outpath=out_dir / "train_perplexity_plot.png",
    )

    plot_table(
        valid_table,
        title="Validation Perplexity over Epochs",
        ylabel="Validation Perplexity",
        outpath=out_dir / "valid_perplexity_plot.png",
    )

    print("Done. Files saved in 'analysis' directory.")
    print(train_table)
    print(valid_table)
    print(test_table)


if __name__ == "__main__":
    main()