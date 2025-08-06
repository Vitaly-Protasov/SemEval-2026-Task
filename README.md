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