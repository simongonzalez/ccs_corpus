import pandas as pd
import stanza
import re
from itertools import groupby
from collections import Counter
from docx import Document
from subprocess import call
import os
import pypandoc

# Initialize Stanza
stanza.download('es')  # Download Spanish models
nlp = stanza.Pipeline('es')


# Define helper functions
def clean_text(text):
    text = re.sub(r'\s*\[[^\]]+\]', '', text)  # Remove text between []
    text = re.sub(r'Habl\..*?|Enc\.[0-9]\:', '', text)  # Remove speaker indicators
    text = re.sub(r'["\'/]', '', text)  # Remove certain punctuation
    text = text.lower().strip()
    return text


def get_question_structure(text):
    if '¿' in text and '?' in text:
        return 'Inline'
    elif '¿' in text:
        return 'Start'
    elif '?' in text:
        return 'End'
    return 'Other'


def get_exclamation_structure(text):
    if '¡' in text and '!' in text:
        return 'Inline'
    elif '¡' in text:
        return 'Start'
    elif '!' in text:
        return 'End'
    return 'Other'


def read_doc(file_path):
    doc = Document(file_path)
    lines_all = []
    for paragraph in doc.paragraphs:
        # Split the text of the paragraph into lines
        lines = paragraph.text.split('\n')
        # Iterate through each line in the paragraph
        for line in lines:
            lines_all.append(line)
    return lines_all


def split_text(row):
    pattern = r'\.\.\.|/TEXT/|//|\.|/|;'
    split_texts = re.split(pattern, row['text'])
    # Remove empty strings
    split_texts = re.split(pattern, row['text'])
    split_texts = [text for text in split_texts if text.strip() != '']
    return pd.Series(split_texts)


def get_rleid(series):
    return pd.Series((k for k, _ in groupby(series)))


def not_in(a, b):
    return np.array([x not in b for x in a])


def convert_to_docx_if_needed(input_file):
    output_file = os.path.splitext(input_file)[0] + '.docx'

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping conversion.")
    else:
        try:
            pypandoc.convert_file(input_file, 'docx', outputfile=output_file)
            print(f"File converted and saved to {output_file}")
        except RuntimeError:
            output_path = os.path.dirname(input_file)
            call(['libreoffice', '--convert-to', 'docx', input_file, '--outdir', output_path])
            print(f"File converted and saved to {output_file}")


def annotate_text(text):
    return nlp(text)

def process_docx(file_path, output_folder):
    # Read the content of the .docx file
    raw_lines = read_doc(file_path)
    raw_lines = [line.strip() for line in raw_lines if line.strip()]

    df_doc = pd.DataFrame({'raw': raw_lines})

    # Assign speaker labels
    df_doc['speaker'] = df_doc['raw'].apply(
        lambda x: 'Hablante' if x.startswith('Habl') else ('Entrevistador' if re.match(r'^Enc\.[0-9]:', x) else None))
    df_doc['speaker'] = df_doc['speaker'].ffill()
    df_doc['text'] = df_doc['raw'].str.replace(
        r'Habl.:|Habl.|Habl:|Enc.\d:|Enc.:|Enc.|HABL:|I\.:|E\.:|E[0-9]:|AUX1:|O:', ''
        , regex=True)

    df_doc = df_doc.apply(split_text, axis=1).stack().reset_index(level=1, drop=True).to_frame('text').join(
        df_doc[['speaker', 'raw']], how='left')

    # Apply clean_text function
    df_doc['text'] = df_doc['text'].apply(clean_text)

    # Clean the split text
    df_doc = df_doc.dropna(subset=['speaker', 'text']).reset_index(drop=True)

    df_doc['speaker'] = df_doc['speaker'].fillna(method='ffill')

    df_doc['text_clean'] = df_doc['text'].apply(lambda x: re.sub(r'[,\.?¿\.\.\.!¡]', '', x).strip())
    df_doc = df_doc[df_doc['text_clean'].apply(lambda x: x not in ['', ' '])]

    df_doc = df_doc.reset_index(drop=True)
    df_doc = df_doc.reset_index(drop=True)
    df_doc['group_turn'] = (df_doc['speaker'] != df_doc['speaker'].shift()).cumsum()
    df_doc['speakerGroup_turn'] = df_doc.groupby('speaker')['group_turn'].transform(lambda x: (x != x.shift()).cumsum())
    df_doc['line_number_n'] = df_doc.index + 1
    df_doc['doc_index'] = df_doc.index + 1
    df_doc['doc_id'] = 'doc' + df_doc['doc_index'].astype(str)

    # Identify repetitions between turns
    df_doc['repeat_nextTurn'] = 0
    df_doc['repeat_nextTurn_word'] = ''

    for i in range(len(df_doc) - 1):
        last_current_word = df_doc['text'].iloc[i].split()[-1]
        first_next_word = df_doc['text'].iloc[i + 1].split()[0]

        if last_current_word == first_next_word:
            df_doc.loc[i, 'repeat_nextTurn'] = 1
            df_doc.loc[i, 'repeat_nextTurn_word'] = last_current_word

    df_doc['stanza'] = df_doc['text'].apply(annotate_text)

    # Convert Stanza annotations to DataFrame format
    annotations = []

    for i, doc in df_doc.iterrows():
        for sentence in doc['stanza'].sentences:
            for word in sentence.words:
                annotations.append({
                    'doc_id': doc['doc_id'],
                    'token': word.text,
                    'paragraph_id': sentence.index,
                    'sentence_id': word.id,
                    'text': sentence.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'deprel': word.deprel,
                    'head_token_id': word.head,
                    'misc': word.misc
                })
    df_ud = pd.DataFrame(annotations)

    df_ud_matched = df_ud.merge(df_doc, on='doc_id', suffixes=('_ud', ''))
    df_ud_matched['is_questionTag'] = df_ud_matched['text'].str.contains(r'\¿no\?').astype(int)
    df_ud_matched['question_structure'] = df_ud_matched.apply(get_question_structure, axis=1)
    df_ud_matched['exclamation_structure'] = df_ud_matched.apply(get_exclamation_structure, axis=1)

    for i in range(len(df_ud_matched) - 1):
        if (df_ud_matched.loc[i, 'question_structure'] == 'Start' and
                df_ud_matched.loc[i + 1, 'question_structure'] == 'Other'):
            df_ud_matched.loc[i + 1, 'question_structure'] = 'Start'
        if df_ud_matched.loc[i, 'exclamation_structure'] == 'Start' and df_ud_matched.loc[
            i + 1, 'exclamation_structure'] == 'Other':
            df_ud_matched.loc[i + 1, 'exclamation_structure'] = 'Start'

    # Match tagging
    df_ud_matched['is_question'] = df_ud_matched['question_structure'].isin(['Inline', 'Start', 'End']).astype(int)
    df_ud_matched['is_exclamation'] = df_ud_matched['exclamation_structure'].isin(['Inline', 'Start', 'End']).astype(
        int)
    df_ud_matched['is_statement'] = (
                (df_ud_matched['is_question'] == 0) & (df_ud_matched['is_exclamation'] == 0)).astype(int)

    # Calculate statistics
    df_ud_matched['char_count'] = df_ud_matched['text'].str.len()
    df_ud_matched['word_count'] = df_ud_matched['text'].str.split().str.len()

    # Overall stats
    df_ud_matched['total_Groupturns'] = df_ud_matched['group_turn'].max()
    df_ud_matched['total_turns'] = len(df_ud_matched)
    df_ud_matched['total_chars'] = df_ud_matched['char_count'].sum()
    df_ud_matched['total_words'] = df_ud_matched['word_count'].sum()
    df_ud_matched['average_chars'] = df_ud_matched['char_count'].mean()
    df_ud_matched['average_words'] = df_ud_matched['word_count'].mean()

    # Sentence types
    df_ud_matched['total_questionTags'] = df_ud_matched['is_questionTag'].sum()
    df_ud_matched['total_questions'] = df_ud_matched['is_question'].sum()
    df_ud_matched['total_exclamations'] = df_ud_matched['is_exclamation'].sum()
    df_ud_matched['total_statements'] = df_ud_matched['is_statement'].sum()

    # Repetitions
    df_ud_matched['total_repetitions'] = df_ud_matched['repeat_nextTurn'].sum()

    # Stats by speaker
    speaker_stats = df_ud_matched.groupby('speaker').agg({
        'speakerGroup_turn': 'max',
        'doc_id': 'count',
        'char_count': 'sum',
        'word_count': 'sum',
        'is_questionTag': 'sum',
        'is_question': 'sum',
        'is_exclamation': 'sum',
        'is_statement': 'sum',
        'repeat_nextTurn': 'sum'
    }).rename(columns={
        'speakerGroup_turn': 'speaker_total_Groupturns',
        'doc_id': 'speaker_total_turns',
        'char_count': 'speaker_total_chars',
        'word_count': 'speaker_total_words',
        'is_questionTag': 'speaker_total_questionTags',
        'is_question': 'speaker_total_questions',
        'is_exclamation': 'speaker_total_exclamations',
        'is_statement': 'speaker_total_statements',
        'repeat_nextTurn': 'speaker_total_repetitions'
    })

    speaker_stats['speaker_average_chars'] = speaker_stats['speaker_total_chars'] / speaker_stats['speaker_total_turns']
    speaker_stats['speaker_average_words'] = speaker_stats['speaker_total_words'] / speaker_stats['speaker_total_turns']

    df_ud_matched = pd.merge(df_ud_matched, speaker_stats, on='speaker')

    # Calculate percentages
    total_turns = df_ud_matched['total_turns'].iloc[0]
    df_ud_matched['speaker_percentage_Groupturns'] = df_ud_matched['speaker_total_Groupturns'] / df_ud_matched[
        'total_Groupturns']
    df_ud_matched['speaker_percentage_turns'] = df_ud_matched['speaker_total_turns'] / total_turns
    df_ud_matched['speaker_percentage_chars'] = df_ud_matched['speaker_total_chars'] / df_ud_matched['total_chars']
    df_ud_matched['speaker_percentage_words'] = df_ud_matched['speaker_total_words'] / df_ud_matched['total_words']
    df_ud_matched['speaker_percentage_questionTags'] = df_ud_matched['speaker_total_questionTags'] / total_turns
    df_ud_matched['speaker_percentage_questions'] = df_ud_matched['speaker_total_questions'] / total_turns
    df_ud_matched['speaker_percentage_exclamations'] = df_ud_matched['speaker_total_exclamations'] / total_turns
    df_ud_matched['speaker_percentage_statements'] = df_ud_matched['speaker_total_statements'] / total_turns
    df_ud_matched['speaker_percentage_repetitions'] = df_ud_matched['speaker_total_repetitions'] / total_turns

    # Intra speaker percentages
    df_ud_matched['intra_speaker_percentage_questionTags'] = df_ud_matched['speaker_total_questionTags'] / \
                                                             df_ud_matched['speaker_total_turns']
    df_ud_matched['intra_speaker_percentage_questions'] = df_ud_matched['speaker_total_questions'] / df_ud_matched[
        'speaker_total_turns']
    df_ud_matched['intra_speaker_percentage_exclamations'] = df_ud_matched['speaker_total_exclamations'] / \
                                                             df_ud_matched['speaker_total_turns']
    df_ud_matched['intra_speaker_percentage_statements'] = df_ud_matched['speaker_total_statements'] / df_ud_matched[
        'speaker_total_turns']
    # Save the results for this document

    df_ud_matched.pop('stanza')

    output_csv = os.path.basename(file_path).split('.')[0]+'.csv'
    output_path = os.path.join(output_folder, output_csv)

    df_ud_matched.to_csv(output_path, index=False)
    print(f"Processed and saved results for {output_path}")


def process_folder(input_folder, output_folder):
    # Convert .doc files to .docx if needed
    for filename in os.listdir(input_folder):
        if filename.endswith('.doc'):
            input_file = os.path.join(input_folder, filename)
            convert_to_docx_if_needed(input_file)

    # Process all .docx files
    for filename in os.listdir(input_folder):
        if filename.endswith('.docx'):
            file_path = os.path.join(input_folder, filename)
            process_docx(file_path, output_folder)

def get_output_directory_path():
    # Get the directory of the current Python file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the output directory path
    output_dir = os.path.join(current_dir, 'Input_Output', 'Output')

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created at: {output_dir}")
    else:
        print(f"Output directory already exists at: {output_dir}")

    return output_dir

# Example usage
if __name__ == '__main__':
    input_folder = '/Users/pradeepchandran/ccs_corpus/Summer2024/01_Microservices/1_TranscriptionMapping/Input_Output/Input'
    output_folder = get_output_directory_path()
    process_folder(input_folder, output_folder)