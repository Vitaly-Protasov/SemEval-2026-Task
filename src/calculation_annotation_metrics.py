import pandas as pd
import numpy as np
import itertools
import ast
from typing import Iterable
import re

from src.utils.xml_parser import parse_xml_file_to_dataframe

train_dict_paths = {
    "annotations": [
        "/home/jovyan/airi/semeval-2026/annotation_quality/train/16-07-2025-train_NL+IS.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/train/16-07-2025-train_Oksana.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/train/16-07-2025-train_Tkachenko.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/train/16-07-2025-train-Lapanicyna.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/train/Анастасия_train_utf8.xml",
    ],
    "merged": "/home/jovyan/airi/semeval-2026/annotation_quality/train/merged_df.xml",
    "final": "/home/jovyan/airi/semeval-2026/annotation_quality/train/final_train.xml",
}

test_dict_paths = {
    "annotations": [
        "/home/jovyan/airi/semeval-2026/annotation_quality/test/Anastasia-test.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/test/Ксения-test-25.08.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/test/Мария-test.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/test/Наталья-test.xml",
        "/home/jovyan/airi/semeval-2026/annotation_quality/test/Олеся-test.xml",
    ],
    "merged": "/home/jovyan/airi/semeval-2026/annotation_quality/test/test_merged.csv",
    "final": "/home/jovyan/airi/semeval-2026/annotation_quality/test/test_final (1).xml",
}


def _parse(col: pd.Series) -> list[list]:
    """Parse stringified lists OR pass through real lists."""

    def _safe_parse(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        elif isinstance(x, list):
            return x
        else:
            return []

    return col.apply(_safe_parse).tolist()


def _f1_score(
    gold: Iterable[tuple],
    pred: Iterable[tuple],
) -> float:
    """Compute F1 score given gold and predicted tuples."""
    gold_set = set(gold)
    pred_set = set(pred)

    correct = len(gold_set & pred_set)
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_f1(df: pd.DataFrame) -> dict[str, float] | None:
    if df is None or df.empty:
        return None

    if df.isnull().values.any():
        print("Warning: DataFrame contains NaN values.")

    # Parse columns once
    a_aspect = _parse(df["Annotator_Aspect"])
    a_opinion = _parse(df["Annotator_Opinion"])
    a_category = _parse(df["Annotator_Category"])

    r_aspect = _parse(df["Reviewed_Aspect"])
    r_opinion = _parse(df["Reviewed_Opinion"])
    r_category = _parse(df["Reviewed_Category"])

    gold_aoc, pred_aoc = [], []
    gold_ac, pred_ac = [], []
    gold_a, pred_a = [], []

    for aa, ao, ac, ra, ro, rc in zip(
        a_aspect,
        a_opinion,
        a_category,
        r_aspect,
        r_opinion,
        r_category,
    ):
        gold_aoc.extend(zip(aa, ao, ac))
        pred_aoc.extend(zip(ra, ro, rc))

        gold_ac.extend(zip(aa, ac))
        pred_ac.extend(zip(ra, rc))

        gold_a.extend([(a,) for a in aa])
        pred_a.extend([(a,) for a in ra])

    return {
        "f1_aoc": _f1_score(gold_aoc, pred_aoc),
        "f1_ac": _f1_score(gold_ac, pred_ac),
        "f1_a": _f1_score(gold_a, pred_a),
    }


def calculate_rmse(df):
    if df is None:
        return None
    if isinstance(df, list):
        df = pd.DataFrame(df)
    if df.empty or df.shape[1] < 2:
        return None
    # Convert to numeric, handle non-numeric values as NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isnull().values.any():
        print("Warning: DataFrame contains NaN values.")

    rmse_squared_errors = []
    for _, row in df.iterrows():
        values = row.values
        if len(values) >= 2:
            # Generate all possible pairs of annotators
            for i, j in itertools.combinations(range(len(values)), 2):
                # Skip pairs with missing values
                if not (np.isnan(values[i]) or np.isnan(values[j])):
                    # Calculate absolute difference between annotator ratings
                    squared_error = (values[i] - values[j]) ** 2
                    rmse_squared_errors.append(squared_error)
    if not rmse_squared_errors:
        return None
    mean_squared_error = np.mean(rmse_squared_errors)

    rmse = np.sqrt(mean_squared_error)
    # Return mean of all absolute differences
    return float(rmse)


def _read_file_to_df(path):
    if path.endswith(".xml"):
        return parse_xml_file_to_dataframe(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)


def get_mean_res(stage_scores):
    return {k: float(np.mean([s[k] for s in stage_scores])) for k in stage_scores[0]}


def main_f1():
    # train dataframes
    train_annotation_dfs = []
    for path in train_dict_paths["annotations"]:
        train_annotation_dfs.append(_read_file_to_df(path))
    df_train_final = _read_file_to_df(train_dict_paths["final"])

    # test dataframes
    test_annotation_dfs = []
    for path in test_dict_paths["annotations"]:
        test_annotation_dfs.append(_read_file_to_df(path))
    df_test_final = _read_file_to_df(test_dict_paths["final"])

    all_f1_scores = []
    for stage_annotation_dfs, stage_final_df in zip(
        [train_annotation_dfs, test_annotation_dfs], [df_train_final, df_test_final]
    ):
        annotation_dfs = []
        for _df in stage_annotation_dfs:
            dict_merge = {
                "Annotator_Aspect": [],
                "Annotator_Opinion": [],
                "Annotator_Category": [],
                "Reviewed_Aspect": [],
                "Reviewed_Opinion": [],
                "Reviewed_Category": [],
            }

            for text in stage_final_df["text"].unique():
                _df_text = _df[_df["text"] == text]
                _df_merged = stage_final_df[stage_final_df["text"] == text]

                if not len(_df_merged) or not len(_df_text):
                    continue

                Annotator_Aspect_temp = []
                Annotator_Opinion_temp = []
                Annotator_Category_temp = []
                Reviewed_Aspect_temp = []
                Reviewed_Opinion_temp = []
                Reviewed_Category_temp = []

                for _, sub_row in _df_text.iterrows():
                    ann_target = sub_row["target"]
                    ann_opinion = sub_row["opinion"]
                    ann_category = sub_row["category"]

                    Annotator_Aspect_temp.append(ann_target.lower().strip())
                    Annotator_Opinion_temp.append(ann_opinion.lower().strip())
                    Annotator_Category_temp.append(ann_category.lower().strip())

                for _, row in _df_merged.iterrows():
                    Reviewed_Aspect_temp.append(row["target"].lower().strip())
                    Reviewed_Opinion_temp.append(row["opinion"].lower().strip())
                    Reviewed_Category_temp.append(row["category"].lower().strip())

                dict_merge["Annotator_Aspect"].append(Annotator_Aspect_temp)
                dict_merge["Annotator_Opinion"].append(Annotator_Opinion_temp)
                dict_merge["Annotator_Category"].append(Annotator_Category_temp)
                dict_merge["Reviewed_Aspect"].append(Reviewed_Aspect_temp)
                dict_merge["Reviewed_Opinion"].append(Reviewed_Opinion_temp)
                dict_merge["Reviewed_Category"].append(Reviewed_Category_temp)

            annotation_dfs.append(pd.DataFrame(dict_merge))

        stage_scores = [calculate_f1(df) for df in annotation_dfs if calculate_f1(df)]

        all_f1_scores.append(get_mean_res(stage_scores))

    print("Mean F1=", get_mean_res(all_f1_scores))


def main_rmse():
    # train dataframes
    train_annotation_dfs = []
    for path in train_dict_paths["annotations"]:
        train_annotation_dfs.append(_read_file_to_df(path))
    df_train_merged = _read_file_to_df(train_dict_paths["merged"])

    # test dataframes
    test_annotation_dfs = []
    for path in test_dict_paths["annotations"]:
        test_annotation_dfs.append(_read_file_to_df(path))
    df_test_merged = _read_file_to_df(test_dict_paths["merged"])

    all_rmse_scores = []
    for stage_annotation_dfs, stage_merged_df in zip(
        [train_annotation_dfs, test_annotation_dfs], [df_train_merged, df_test_merged]
    ):
        rmse_dict = {
            "annotator_0": [],
            "annotator_1": [],
            "annotator_2": [],
            "annotator_3": [],
            "annotator_4": [],
        }

        for _, row in stage_merged_df.iterrows():
            text = row["text"].strip()
            target = row["target"]

            if "???" in row["opinion"]:
                match = re.search(r"\(([^)]*)\)", row["opinion"])
                opinion_list = match.group(1).split(";") if match else []
            else:
                opinion_list = [row["opinion"]]

            for ann_id, sub_df in enumerate(stage_annotation_dfs):
                _df_an = sub_df[
                    (sub_df["text"] == text)
                    & (sub_df["target"] == target)
                    & (sub_df["opinion"].isin(opinion_list))
                ]
                if len(_df_an):
                    rmse_dict[f"annotator_{ann_id}"].append(_df_an.iloc[0].intensity)
                else:
                    rmse_dict[f"annotator_{ann_id}"].append(None)

        df_intensity = pd.DataFrame(rmse_dict)

        df_first = df_intensity.applymap(
            lambda x: float(x.split("#")[0]) if isinstance(x, str) else None
        )

        df_second = df_intensity.applymap(
            lambda x: float(x.split("#")[1]) if isinstance(x, str) else None
        )
        all_rmse_scores.append((calculate_rmse(df_first), calculate_rmse(df_second)))

    print("Mean RMSE=", np.mean(all_rmse_scores, axis=0))


if __name__ == "__main__":
    main_f1()
    main_rmse()
