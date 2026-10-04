from pathlib import Path

from joblib import parallel_backend

if __package__:
    from .history_experiment import HistoryExperiment
    from .models import GradientBoosting
else:
    from history_experiment import HistoryExperiment
    from models import GradientBoosting

OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "boosting_history_experiment"


def main():
    baseline = GradientBoosting()
    history = GradientBoosting()
    baseline.model.set_params(n_jobs=2)
    history.model.set_params(n_jobs=2)
    models = [
        ("baseline", baseline, False),
        ("history", history, True),
    ]
    with parallel_backend("threading"):
        experiment = HistoryExperiment(models, OUTPUT)
        experiment.run()
    (OUTPUT / "experiment_notes.txt").write_text(
        "GradientBoostingRegressor: baseline 12 raw features versus 12 raw + 77 causal history features.\n"
        "Same target, engine split 1-80 / 81-100, GroupKFold(5), and parameter grid.\n"
        "Grid: n_estimators=[100,200], learning_rate=[0.05,0.1], max_depth=[2,3].\n"
        "Fixed min_samples_leaf=10, random_state=42; no scaling or polynomial expansion.\n"
        "History windows include current and past samples only; "
        "initial baseline uses up to first 25 observed cycles.\n"
        "Validation does not participate in parameter selection. Metrics use unsmoothed predictions.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
