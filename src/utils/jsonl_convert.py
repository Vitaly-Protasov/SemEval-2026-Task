import pandas as pd

from src.utils.models import (
    SemEvalFormatLineModel,
    SemEvalFormatQuadrupletModel,
    XMLFieldsFinal,
)


def convert_df_to_jsonl(
    df: pd.DataFrame, jsonl_path: str, sep_intensity: str = "#", sep_std: str = "±"
):
    ids = []
    texts = []
    quadriplets = []

    for i, text in enumerate(df.text.unique()):
        sub_df = df[df[XMLFieldsFinal.TEXT] == text]

        quads: list[SemEvalFormatQuadrupletModel] = []
        for _, row in sub_df.iterrows():
            opinion = row.opinion.replace("???", "")

            va = sep_intensity.join(
                [i.split(sep_std)[0] for i in row.intensity.split(sep_intensity)]
            )

            quads.append(
                SemEvalFormatQuadrupletModel(
                    Aspect=row.target, Opinion=opinion, Category=row.category, VA=va
                )
            )

        ids.append(f"{row.sentence_id}_{i}")
        texts.append(text)
        quadriplets.append(quads)
    data = SemEvalFormatLineModel(
        ID=ids,
        Text=texts,
        Quadruplet=[[q.dict() for q in q_inner_list] for q_inner_list in quadriplets],
    )
    final_df = pd.DataFrame(data.dict())
    final_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
