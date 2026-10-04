from pathlib import Path

if __package__:
    from .history_experiment import HistoryExperiment
    from .models import HistoryPolynomialRidge, PolynomialRidge
else:
    from history_experiment import HistoryExperiment
    from models import HistoryPolynomialRidge, PolynomialRidge

OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "sensor_history_experiment"


def main(models=None, output_directory=OUTPUT):
    default_models = models is None
    if models is None:
        models = [
            ("baseline", PolynomialRidge(), False),
            ("history", HistoryPolynomialRidge(), True),
        ]
    experiment = HistoryExperiment(models, output_directory)
    experiment.run()
    if default_models:
        notes = (
            "Polynomial degree 3 on the same 12 inputs; history adds 77 linear features.\n"
            "Feature blocks: 454 original polynomial features + 77 history features = 531.\n"
            "Same engine split 1-80 / 81-100, GroupKFold(5), and 8 alpha candidates.\n"
            "History: rolling mean/std/OLS slopes over 10 and 30 observations; "
            "delta from causal initial mean of up to 25 observations.\n"
            "Only current and past observations are used. Targets and indices are preserved.\n"
            "Scalers fit inside CV; validation is excluded from alpha selection.\n"
            "Metrics use unsmoothed predictions. Active model configuration is unchanged.\n"
        )
        Path(output_directory, "experiment_notes.txt").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
