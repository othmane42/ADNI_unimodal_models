
from typing import Callable


def get_serialize_data_processing() -> Callable:

    # Mapping of column names to more understandable descriptions
    column_descriptions = {
        # Categorical columns
        "PTRACCAT": "Race",
        "PTGENDER": "Gender",
        "PTHAND": "Handedness",
        "PTMARRY": "Marital Status",
        "PTNOTRT": "Participant Retired",
        "PTTLANG": "Primary Language",
        "MH14ALCH": "Alcohol Use",
        "MH17MALI": "Malignancy History",
        "MH16SMOK": "Smoking History",
        "MH15DRUG": " Drug Abuse",
        "MH4CARD": "Cardiovascular History",
        "MHPSYCH": "Psychiatric History",
        "MH2NEURL": "Neurologic (other than AD) History",
        "MH6HEPAT": "Hepatic History",
        "MH12RENA": "Renal-Genitourinary History",
        "DSPANFOR": "DSpan Forward",
        "DSPANBAC": "DSpan Backward",
        "CDGLOBAL": "CDR global score",
        "BCFAQ": "Functional Assessment Questionnaire",
        "BCDEPRES": "Depression Assessment",

        # Continuous columns
        "VSWEIGHT": "Weight",
        "VSHEIGHT": "Height",
        "MMSCORE": "Mini-Mental State Examination Score",
        "TRAASCOR": "Time to Complete part A",
        "TRABSCOR": "Time to Complete part B",
        "TRABERRCOM": "Errors of Commission",
        "CATANIMSC": "Category Fluency Animal Score",
        "BNTTOTAL": "Total Number Correct (1+3)",
        "PTEDUCAT": "Years of Education",
        "PTDOBYY": "Year of Birth"
    }

    # Mapping of codes to descriptions for categorical variables
    code_mappings = {
        "PTRACCAT": {'1': "American Indian or Alaskan Native", '2': "Asian", '3': "Native Hawaiian or Other Pacific Islander",
                     '4': "Black or African American", '5': "White", '6': "More than one race", '7': "Unknown",
                     '8': "Native Hawaiian", '9': "Other Pacific Islander"},
        "PTGENDER": {1: "Male", 2: "Female"},
        "PTHAND": {1: "Right", 2: "Left"},
        "PTMARRY": {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Never married", 5: "Unknown", 6: "Domestic Partnership"},
        "PTNOTRT": {0: "No", 1: "Yes", 2: "Not Applicable"},
        "PTTLANG": {1: "English", 2: "Other"},
        "MH14ALCH": {0: "No", 1: "Yes"},
        "MH17MALI": {0: "No", 1: "Yes"},
        "MH16SMOK": {0: "No", 1: "Yes"},
        "MH15DRUG": {0: "No", 1: "Yes"},
        "MH4CARD": {0: "No", 1: "Yes"},
        "MHPSYCH": {0: "No", 1: "Yes"},
        "MH2NEURL": {0: "No", 1: "Yes"},
        "MH6HEPAT": {0: "No", 1: "Yes"},
        "MH12RENA": {0: "No", 1: "Yes"},
        "BCFAQ": {0: "No", 1: "Yes"},
        "BCDEPRES": {0: "No", 1: "Yes"}
    }

    prompt = ""

    def serialize_data(df):
        def serialize_row(row):
            row_dict = row.to_dict()
            # Serialize data with inverse transformation
            row_text = ", ".join(
                f"{column_descriptions.get(key, key)}: {code_mappings.get(key, {}).get(row_dict[key], f'{row_dict[key]:.2f}' if isinstance(row_dict[key], (int, float)) else row_dict[key])}"
                for key in row_dict
                if key in column_descriptions  # Only include columns that are in our descriptions
            )

            return prompt + row_text

        serialized_data = df.apply(
            lambda row: serialize_row(row),
            axis=1
        ).tolist()

        return serialized_data

    return serialize_data
