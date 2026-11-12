from config import DATASETS, ABODA


def check_path(label, path):
    if path.exists():
        print(f"[PASS] {label}")
        return True

    print(f"[FAIL] {label}: {path}")
    return False


def main():
    passed = True

    for key, dataset in DATASETS.items():
        passed &= check_path(
            f"{dataset['name']} root",
            dataset["root"]
        )

        if "train" in dataset:
            splits = ["train", "test"]

            for split in splits:
                for role in ["t0", "t1", "mask"]:
                    path = dataset["root"] / dataset[split][role]
                    passed &= check_path(
                        f"{dataset['name']} {split}/{role}",
                        path
                    )
        else:
            for role in ["t0", "t1", "mask"]:
                path = dataset["root"] / dataset[role]
                passed &= check_path(
                    f"{dataset['name']} {role}",
                    path
                )

    passed &= check_path(
        "ABODA root",
        ABODA["root"]
    )

    print()

    if passed:
        print("STEP 4B: PASS")
    else:
        print("STEP 4B: FAIL")


if __name__ == "__main__":
    main()