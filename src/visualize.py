import argparse

from evaluate_scores import plot_eval_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "folder_name",
        nargs="?",
        default="Dmel",
        help="Folder name under data/ (default: Dmel)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Show the plot interactively instead of saving",
    )
    args = ap.parse_args()

    plot_eval_scores(species=args.folder_name, interactive=args.interactive)


if __name__ == "__main__":
    main()
