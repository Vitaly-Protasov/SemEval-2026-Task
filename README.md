# SemEval-2026-Task

## Convert XML to pd.DataFrame
```python
from src.utils.xml_parser import parse_xml_file_to_dataframe

df = parse_xml_file_to_dataframe('path to file .xml')
```

## Convert pd.DataFrame to XML
```python
from src.utils.xml_parser import convert_df_to_xml

convert_df_to_xml(df=df, output_file_path="final.xml")
```

## Merge different annotations by opinions and their intensities
```python
from src.utils.xml_merge import merge_annotations

merged_df = merge_annotations(
    annotation_dfs = [df1,df2,df3,df4,df5]
)

# to see ignored texts:
merge_annotations(
    annotation_dfs = [df1,df2,df3,df4,df5]
    verbose=True
)
```

## Convert pd.DataFrame to Jsonl based on the SemEval2026 Format
```python
from src.utils.jsonl_convert import convert_df_to_jsonl

convert_df_to_jsonl(
    df=df,
    jsonl_path="your path.jsonl",
)
