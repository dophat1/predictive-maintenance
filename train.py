from pathlib import Path
from src.data import download_dataset, load_dataset
from src.evaluate import run_evaluation

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary = run_evaluation(load_dataset(download_dataset(root / "data" / "raw")), root / "artifacts")
    print(f"Best model: {summary['best_model']}")
    print(f"Holdout PR-AUC: {summary['holdout']['pr_auc']:.3f}")
    print(f"Holdout recall: {summary['holdout']['recall']:.3f}")
    print(f"Event lead time: {summary['event_detection']['lead_hours']} hours")
