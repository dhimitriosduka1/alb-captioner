import json as j
import pandas as pd
from tqdm import tqdm
from translate import EnToSqTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to the Flickr30k annotations file
FLICKER30K_ANNOTATIONS_PATH = "dataset/flickr30k/results.csv"

# Create an instance of the EnToSqTranslator class
translator = EnToSqTranslator()

# Load the Flickr30k annotations file
annotations = pd.read_csv(FLICKER30K_ANNOTATIONS_PATH, delimiter="|")

# Rename the columns to remove any leading or trailing whitespaces
annotations.columns = annotations.columns.str.strip()

# Define a dictionary to store the Albanian translations
translated_annotations = {}

# Function to translate a single annotation
def translate_annotation(row):
    return row["image_name"], translator.translate(annotation = row["comment"])

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(translate_annotation, row): row for _, row in annotations.iterrows()}

    for future in tqdm(as_completed(futures), total=len(futures)):
        image_name, translation = future.result()
        if image_name not in translated_annotations:
            translated_annotations[image_name] = []
        translated_annotations[image_name].append(translation)

# Save the translated annotations to a JSON file
with open("dataset/flickr30k/translated_annotations.json", "w", encoding='utf-8') as file:
    j.dump(translated_annotations, file, ensure_ascii=False, indent=4)