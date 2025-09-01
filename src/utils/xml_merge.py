import statistics
from collections import Counter

import pandas as pd

from src.utils.models import XMLFieldsFinal, XMLModelFinal


def intensity_format(num: int | float) -> str:
    """Format a number for display, removing unnecessary decimal places.

    Args:
        num: The number to format (int or float)

    Returns:
        Formatted string representation of the number
    """
    if isinstance(num, int) or num == int(num):
        return str(int(num))

    formatted = f"{num:.2f}".rstrip("0").rstrip(".")
    return formatted


def aggregate_scores(
    data: list[str], intensity_sep: str = "#", std_sep: str = "±"
) -> str:
    """Summarize intensity data by calculating means and standard deviations.

    Args:
        data: List of strings in "X#Y" format
        intensity_sep: Separator between X and Y values
        std_sep: Separator between mean and standard deviation

    Returns:
        Formatted string in "mean±std#mean±std" format
    """
    x_scores: list[float] = []
    y_scores: list[float] = []

    for item in data:
        try:
            x_str, y_str = item.split(intensity_sep)
            x_scores.append(float(x_str))
            y_scores.append(float(y_str))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid data format: {item}") from e

    def calculate_stats(values: list[float]) -> tuple[float, float]:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    mean_x, std_x = calculate_stats(x_scores)
    mean_y, std_y = calculate_stats(y_scores)

    return (
        f"{intensity_format(mean_x)}{std_sep}{intensity_format(std_x)}"
        f"{intensity_sep}"
        f"{intensity_format(mean_y)}{std_sep}{intensity_format(std_y)}"
    )


def extract_weighted_substrings(weighted_opinions: dict[str, int]) -> dict[str, int]:
    """Calculate weights for all substrings based on their occurrences in opinion strings.

    Args:
        data: Dictionary where keys are opinion strings and values are their weights

    Returns:
        Dictionary of substrings with their cumulative weights, sorted by weight in descending order
    """
    substring_weights: dict[str, int] = {}

    for opinion1 in weighted_opinions:
        if opinion1 not in substring_weights:
            substring_weights[opinion1] = 0

        for opinion2 in weighted_opinions:
            if opinion1 in opinion2:
                substring_weights[opinion1] += weighted_opinions[opinion2]
    return dict(
        sorted(substring_weights.items(), key=lambda item: item[1], reverse=True)
    )


def merge_annotations(
    annotation_dfs: list[pd.DataFrame], verbose: bool = False
) -> pd.DataFrame:
    """
    Process multiple annotation DataFrames to extract aggregated opinions.

    Args:
        dfs: Arbitrary number of pandas DataFrames in `list` format.

        verbose: Print debug output if True.

    Returns:
        Aggregated pandas DataFrame.
    """
    if not annotation_dfs:
        raise ValueError("At least one DataFrame must be provided.")

    ignored_lines = []
    n_annotations = len(annotation_dfs)

    final_records: list[XMLModelFinal] = []

    df_ref = annotation_dfs[
        0
    ]  # Use first DF as the base for iterating reviews/sentences/texts
    for review_id in df_ref.review_id.unique():
        df_review = df_ref[df_ref.review_id == review_id]
        for sentence_id in df_review.sentence_id.unique():
            df_sentence = df_review[df_review.sentence_id == sentence_id]
            for text in df_sentence.text.unique():
                if verbose:
                    print(f"\n{text=}")

                df_text = df_sentence[df_sentence.text == text]
                for target in df_text.target.unique():
                    if target in {"NULL", "None"}:
                        continue

                    df_target = df_text[df_text.target == target]
                    for category in df_target.category.unique():
                        dfs_matching = [
                            df[
                                (df.review_id == review_id)
                                & (df.sentence_id == sentence_id)
                                & (df.text == text)
                                & (df.target == target)
                                & (df.category == category)
                            ]
                            for df in annotation_dfs
                        ]
                        df_joined = pd.concat(dfs_matching)

                        opinions_count = Counter(df_joined.opinion.tolist())
                        if verbose:
                            print("Case 1:", f"{target=}", opinions_count)

                        processed_opinions = []

                        # FIRST CASE
                        for opinion, count in opinions_count.items():
                            if count > n_annotations * 0.5:
                                row = df_joined[df_joined.opinion == opinion].iloc[0]
                                intensities = df_joined[
                                    df_joined.opinion == opinion
                                ].intensity.tolist()

                                final_records.append(
                                    XMLModelFinal(
                                        review_id=review_id,
                                        sentence_id=sentence_id,
                                        text=text,
                                        target=target,
                                        category=category,
                                        polarity=row[XMLFieldsFinal.POLARITY],
                                        opinion=opinion,
                                        from_=XMLFieldsFinal.FROM,
                                        to=XMLFieldsFinal.TO,
                                        intensity=aggregate_scores(intensities),
                                    )
                                )
                                processed_opinions.append(opinion)

                        for op in processed_opinions:
                            opinions_count.pop(op, None)

                        # SECOND CASE: substring-based matches
                        substring_opinions = extract_weighted_substrings(opinions_count)
                        if verbose:
                            print("Case 2:", substring_opinions)

                        for new_op, count in substring_opinions.items():
                            if count > n_annotations * 0.5:
                                df_substring = df_joined[
                                    (df_joined.target == target)
                                    & (df_joined.category == category)
                                    & (~df_joined.opinion.isin(processed_opinions))
                                    & (df_joined.opinion.str.contains(new_op, na=False))
                                ]
                                intensities = df_substring.intensity.tolist()
                                opinions_join = ";".join(df_substring.opinion.tolist())
                                opinion_str = f"???{new_op}_({opinions_join})"

                                row = df_substring[df_substring.opinion == new_op].iloc[
                                    0
                                ]
                                final_records.append(
                                    XMLModelFinal(
                                        review_id=review_id,
                                        sentence_id=sentence_id,
                                        text=text,
                                        target=target,
                                        category=category,
                                        polarity=row[XMLFieldsFinal.POLARITY],
                                        opinion=opinion_str,
                                        from_=XMLFieldsFinal.FROM,
                                        to=XMLFieldsFinal.TO,
                                        intensity=aggregate_scores(intensities),
                                    )
                                )
                            else:
                                if verbose:
                                    print(
                                        "Case 3:",
                                        f"{review_id=},{sentence_id=},{target=},{category=},{new_op=},{count=}",
                                    )
                                ignored_lines.append(
                                    f"{review_id=},{sentence_id=},{target=},{category=},{new_op=},{count=}"
                                )

    df = pd.DataFrame([d.dict() for d in final_records])
    return df
