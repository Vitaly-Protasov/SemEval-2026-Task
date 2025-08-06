import xml.etree.ElementTree as ET

import pandas as pd

from src.utils.models import XMLFieldsFinal, XMLModelFinal, XMLRawFields


def parse_xml_file_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parses an XML file from the given file path into a pandas DataFrame.

    Each row in the DataFrame will represent an opinion,
    and it will include information about the review, sentence, and the opinion itself.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()  # Return an empty DataFrame on error
    except ET.ParseError:
        print(f"Error: Could not parse XML from '{file_path}'. Check file format.")
        return pd.DataFrame()

    data: list[XMLModelFinal] = []

    for review in root.findall(XMLRawFields.REVIEW):
        review_id = review.get(XMLRawFields.RID)
        for sentence in review.findall(XMLRawFields.SENTENCES):
            sentence_id = sentence.get(XMLRawFields.ID)
            text = (
                sentence.find(XMLRawFields.TEXT).text
                if sentence.find(XMLRawFields.TEXT) is not None
                else ""
            )

            opinions_element = sentence.find(XMLRawFields.OPINIONS)
            if opinions_element is not None:
                found_opinions = False
                for opinion in opinions_element.findall(XMLRawFields.OPINION):
                    found_opinions = True
                    target = opinion.get(XMLFieldsFinal.TARGET)
                    category = opinion.get(XMLFieldsFinal.CATEGORY)
                    polarity = opinion.get(XMLFieldsFinal.POLARITY)
                    from_ = opinion.get(XMLFieldsFinal.FROM)
                    to_ = opinion.get(XMLFieldsFinal.TO)
                    intensity = opinion.get(XMLFieldsFinal.INTENSITY)
                    opinion = opinion.get(XMLFieldsFinal.OPINION)

                    data.append(
                        XMLModelFinal(
                            review_id=review_id,
                            sentence_id=sentence_id,
                            text=text,
                            target=target,
                            category=category,
                            polarity=polarity,
                            opinion=opinion,
                            from_=from_,
                            to=to_,
                            intensity=intensity,
                        )
                    )
                if not found_opinions and text:
                    # If <Opinions> tag exists but has no <Opinion> children
                    data.append(
                        XMLModelFinal(
                            review_id=review_id, sentence_id=sentence_id, text=text
                        )
                    )
            else:
                # If <Opinions> tag is missing entirely
                data.append(
                    XMLModelFinal(
                        review_id=review_id, sentence_id=sentence_id, text=text
                    )
                )

    df = pd.DataFrame([d.dict() for d in data])
    return df


def convert_df_to_xml(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Converts a pandas DataFrame (with the structure from the parsing function)
    back into an XML file following the original format.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        output_file_path (str): The path where the XML file will be saved.
    """
    if XMLFieldsFinal.FROM_ in df.columns:
        df = df.rename(columns={XMLFieldsFinal.FROM_: XMLFieldsFinal.FROM})

    temp_column1 = "sentence_id_x"
    temp_column2 = "sentence_id_y"
    df[temp_column1] = df[XMLFieldsFinal.SENTENCE_ID].apply(
        lambda x: int(x.split(":")[0])
    )
    df[temp_column2] = df[XMLFieldsFinal.SENTENCE_ID].apply(
        lambda x: int(x.split(":")[1])
    )

    df = df.sort_values(by=[temp_column1, temp_column2])
    # Create the root element <Reviews>
    reviews_root = ET.Element(XMLRawFields.REVIEW)
    # Group by review_id to reconstruct individual reviews
    for rid in df[XMLFieldsFinal.REVIEW_ID].unique():
        review_group = df[df[XMLFieldsFinal.REVIEW_ID] == rid]
        review_element = ET.SubElement(reviews_root, XMLRawFields.REVIEW, rid=str(rid))
        sentences_element = ET.SubElement(
            review_element, XMLRawFields.SENTENCES_ELEMENT
        )

        # Group by sentence_id within each review
        for sid in review_group[XMLFieldsFinal.SENTENCE_ID].unique():
            sentence_group = review_group[
                review_group[XMLFieldsFinal.SENTENCE_ID] == sid
            ]
            # Get the original text for this sentence (it should be consistent within the group)
            text = sentence_group[XMLRawFields.TEXT].iloc[0]

            sentence_element = ET.SubElement(
                sentences_element, XMLRawFields.SENTENCE, id=str(sid)
            )
            text_element = ET.SubElement(sentence_element, XMLRawFields.TEXT)
            text_element.text = text

            opinions_element = ET.SubElement(sentence_element, XMLRawFields.OPINIONS)

            # Add opinions for this sentence
            # Only add opinions if 'target' is not None (meaning there was an opinion in the original XML)
            opinion_rows = sentence_group[sentence_group[XMLFieldsFinal.TARGET].notna()]
            if not opinion_rows.empty:
                for _, opinion_row in opinion_rows.iterrows():
                    opinion_attrs = {}
                    if pd.notna(opinion_row[XMLFieldsFinal.TARGET]):
                        opinion_attrs[XMLFieldsFinal.TARGET] = str(
                            opinion_row[XMLFieldsFinal.TARGET]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.CATEGORY]):
                        opinion_attrs[XMLFieldsFinal.CATEGORY] = str(
                            opinion_row[XMLFieldsFinal.CATEGORY]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.POLARITY]):
                        opinion_attrs[XMLFieldsFinal.POLARITY] = str(
                            opinion_row[XMLFieldsFinal.POLARITY]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.OPINION]):
                        opinion_attrs[XMLFieldsFinal.OPINION] = str(
                            opinion_row[XMLFieldsFinal.OPINION]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.FROM]):
                        opinion_attrs[XMLFieldsFinal.FROM] = str(
                            opinion_row[XMLFieldsFinal.FROM]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.TO]):
                        opinion_attrs[XMLFieldsFinal.TO] = str(
                            opinion_row[XMLFieldsFinal.TO]
                        )
                    if pd.notna(opinion_row[XMLFieldsFinal.INTENSITY]):
                        opinion_attrs[XMLFieldsFinal.INTENSITY] = str(
                            opinion_row[XMLFieldsFinal.INTENSITY]
                        )
                    # 'from' and 'to' attributes are not in the DataFrame, so we cannot reconstruct them accurately
                    # without additional information. We will omit them for now.
                    # If you need them, you would need to store them in your DataFrame during parsing.
                    ET.SubElement(
                        opinions_element, XMLFieldsFinal.OPINION, **opinion_attrs
                    )

    # Create an ElementTree object
    tree = ET.ElementTree(reviews_root)

    # Write to file with pretty printing
    ET.indent(tree, space="    ")  # For pretty printing with 4 spaces
    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)
    print(f"XML data successfully written to '{output_file_path}'")
