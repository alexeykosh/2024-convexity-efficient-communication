import pandas as pd
import string
import pickle
from colour import Lab_to_XYZ, XYZ_to_sRGB
import numpy as np

# Functions:

def clip(rgb):
    """Clip RGB values to the range [0, 1]."""
    return np.clip(rgb, 0, 1)

def get_color_matrix(lg_id, terms):
    """Generate a color matrix for a given language ID."""
    lg_1 = terms[terms['lg'] == lg_id]
    # Exclude terms like '*'
    lg_1 = lg_1[lg_1['term'] != '*']
    # Group by row, column, and term, and get the count of each term
    lg_1 = lg_1.groupby(['row', 'col', 'term']).size().reset_index(name='count')
    # Sort by row, column, and count in descending order
    lg_1 = lg_1.sort_values(['row', 'col', 'count'], ascending=[True, True, False])
    # For each row and column, select the term with the highest count
    lg_1 = lg_1.groupby(['row', 'col']).first().reset_index()
    if lg_1['count'].min() >= 2:
        # Convert terms to integers by enumerating them
        term_map = {term: i for i, term in enumerate(lg_1['term'].unique())}
        lg_1['term'] = lg_1['term'].apply(lambda x: term_map[x])
        # Pivot the data to create the color matrix
        return lg_1.pivot(index='row', columns='col', values='term').fillna(-1).astype(int).values
    return None

# Loading the data:

# Load language information
lang_info = pd.read_csv('data/raw/langs_info.txt', sep='\t', header=None)
# Create a dictionary of language info
lang_info = dict(zip(lang_info[0], lang_info[1]))
# Pickle the dictionary
with open('data/lang_info.pkl', 'wb') as f:
    pickle.dump(lang_info, f)

# Define row labels
ROWS = list(string.ascii_uppercase[:10])

# Load chip information
chips = pd.read_csv('data/raw/chip.txt', sep='\t', header=None, names=['id', 'row', 'col', 'conc'])
# Convert row labels to indices
chips['row'] = chips['row'].apply(lambda x: ROWS.index(x))
# Create a dictionary mapping chip ID to (row, col)
chip_dict = {row['id']: (row['row'], row['col']) for _, row in chips.iterrows()}
# Find chip numbers in column 0
row_0 = [k for k, v in chip_dict.items() if v[1] == 0]

# Load term data
terms = pd.read_csv('data/raw/term.txt', sep='\t', header=None, names=['lg', 'speaker', 'chip', 'term'])
# Remove terms corresponding to chips in column 0
terms = terms[~terms['chip'].isin(row_0)]
# Map chips to their coordinates
terms['row'] = terms['chip'].apply(lambda x: chip_dict[x][0])
terms['col'] = terms['chip'].apply(lambda x: chip_dict[x][1])
# Pickle the chip dictionary
with open('data/chip_dict.pkl', 'wb') as f:
    pickle.dump(chip_dict, f)

# Generate color matrices for each language
lg_color = {lg_id: get_color_matrix(lg_id, terms) for lg_id in terms['lg'].unique()}
# Exclude languages with no color matrix
lg_color = {k: v for k, v in lg_color.items() if v is not None}
# Pickle the language color matrices
with open('data/lg_color.pkl', 'wb') as f:
    pickle.dump(lg_color, f)

# Load CIELAB data
cielab = pd.read_csv('data/raw/cielab.txt', sep='\t', header=0)
# Map chip numbers to coordinates
cielab['row'] = cielab['#cnum'].apply(lambda x: chip_dict[x][0])
cielab['col'] = cielab['#cnum'].apply(lambda x: chip_dict[x][1])
cielab['Lab'] = cielab[['L*', 'a*', 'b*']].values.tolist()
# Convert LAB to RGB
cielab['RGB'] = cielab['Lab'].apply(lambda x: clip(XYZ_to_sRGB(Lab_to_XYZ(x))))

# Create dictionaries for CIELAB and RGB values
cielab_dict = {(row['row'] - 1, row['col'] - 1): (row['L*'], row['a*'], row['b*']) for _, row in cielab.iterrows()}
rgb_dict = {(row['row'], row['col']): row['RGB'] for _, row in cielab.iterrows()}

# Pickle the CIELAB dictionary
with open('data/cielab_dict.pkl', 'wb') as f:
    pickle.dump(cielab_dict, f)

# Pickle the RGB dictionary
with open('data/rgb_dict.pkl', 'wb') as f:
    pickle.dump(rgb_dict, f)
