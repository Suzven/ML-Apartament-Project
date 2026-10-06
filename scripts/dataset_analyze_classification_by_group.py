if __package__:
    from .dataset_analyze_classification import main as analyze_classification
else:
    from dataset_analyze_classification import main as analyze_classification


def main(model=None):
    analyze_classification(model=model, grouped=True)


if __name__ == "__main__":
    main()
