import pandas as pd
import re
import os
from itertools import islice
import parselmouth
from parselmouth.praat import call
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import chardet
from docx import Document
import tgt
from subprocess import call
import os
import pypandoc


def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']


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


def clean_text(text):
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return None


def split_text(row):
    pattern = r'\.\.\.|/TEXT/|//|\.|/|;'
    split_texts = re.split(pattern, row['text'])
    # Remove empty strings
    split_texts = [text for text in split_texts if text.strip() != '']
    return pd.DataFrame({
        'raw': row['raw'],
        'speaker': row['speaker'],
        'text': split_texts
    })


def determine_speaker(text):
    if re.search(r'^(Habl|HABL|I\.|I:|O:)', text):
        return 'Hablante'
    elif re.search(r'^(Enc\.:|Enc\.[0-9]:|E\.|E[0-9]:|AUX[0-9]:)', text):
        return 'Entrevistador'
    else:
        return None


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


def main():
    script_path = os.path.abspath(__file__)
    current_working_directory = os.path.dirname(script_path)
    # Read the CSV file
    duration_file_path = os.path.join(current_working_directory, 'Input_Output', 'Input',
                                      'XX_duration_files_20240324.csv')
    fls_df = pd.read_csv(duration_file_path)

    # List all .doc files in the directory
    manual_transcript_file_path = os.path.join(current_working_directory, 'Input_Output', 'Input')

    fls_doc_original = [os.path.join(manual_transcript_file_path, f) for f in os.listdir(manual_transcript_file_path) if
               f.endswith('.doc')]

    for file in fls_doc_original:
        convert_to_docx_if_needed(file)

    fls_doc = [os.path.join(manual_transcript_file_path, f) for f in os.listdir(manual_transcript_file_path) if
               f.endswith('.docx')]

    #current_working_directory = os.getcwd()

    # List all .srt files in the directory
    srt_file_path = os.path.join(current_working_directory, 'Input_Output', 'Input')
    fls_son = [os.path.join(srt_file_path, f) for f in os.listdir(srt_file_path)
               if f.endswith('.srt')]

    # Create a DataFrame for .srt files
    dfall_son = pd.DataFrame({
        'srt': fls_son,
        'file_srt': [os.path.basename(f) for f in fls_son],
        'name': [re.sub(r'\.mp3\.srt$', '', os.path.basename(f)) for f in fls_son]
    })

    # Create a DataFrame for .doc files and merge with dfall_son and fls_df
    dfall = pd.DataFrame({
        'doc': fls_doc,
        'file_doc': [os.path.basename(f) for f in fls_doc],
        'name': [re.sub(r'\.docx$', '', os.path.basename(f)) for f in fls_doc]
    }).merge(dfall_son, on='name', how='left').merge(fls_df, on='name', how='left').dropna(subset=['srt'])

    output_directory = os.path.join(current_working_directory, 'Input_Output', 'Output', 'XX_TG_Matched')
    os.makedirs(output_directory, exist_ok=True)

    df_doc_output_directory = os.path.join(output_directory, 'df_doc_ngrams')
    os.makedirs(df_doc_output_directory, exist_ok=True)
    for index, file in dfall.iterrows():
        df_doc = pd.DataFrame({'raw': [line.strip()
                                       for line in read_doc(file['doc'])
                                       if line.strip()]})

        pattern = r'\.\.\.|/TEXT/|//|\.|/|;'
        # create a speaker column and fill it downwards
        # df_doc['speaker'] = df_doc['raw'].apply(lambda x: 'Hablante' if re.match(r'^Habl', x)
        # else ('Entrevistador' if re.match(r'^Enc\.\d:', x) else None))
        df_doc['speaker'] = df_doc['raw'].apply(determine_speaker)
        df_doc['speaker'] = df_doc['speaker'].fillna(method='ffill')
        df_doc = df_doc.dropna(subset=['speaker'])

        # clean the text column
        # df_doc['text'] = df_doc['raw'].str.replace(r'Habl.:|Habl.|Habl:|Enc.\d:|Enc.', '', regex=True).str.strip().str.lower()
        df_doc['text'] = df_doc['raw'].str.replace(r'Habl.:|Habl.|Habl:|Enc.\d:|Enc.:|Enc.|HABL:|I\.:|E\.:|E[0-9]:|AUX1:|O:', ''
                                                   , regex=True)
        df_doc_temp = pd.concat(df_doc.apply(split_text, axis=1).values)
        df_doc = df_doc_temp
        df_doc['text'] = df_doc['text'].str.lower()
        df_doc['text'] = df_doc['text'].str.replace(r'[^\w\s]', '', regex=True)
        df_doc['text'] = df_doc['text'].str.replace(r" pa' ", ' para ', regex=False)
        df_doc['text'] = df_doc['text'].str.replace(r"nadien", 'nadie', regex=False)
        df_doc = df_doc.dropna(subset=['text'])
        df_doc = df_doc[df_doc['text'].str.strip() != '']
        # df_doc['text'] = df_doc['text'].str.replace(r'[,\.\?\¿¡\[\]\"]|\.{3}', '', regex=True)

        df_doc['line_number_n'] = range(1, len(df_doc) + 1)


        print(df_doc)

        # generate ngrams from the manual transcription
        def generate_ngrams(text, n):
            words = word_tokenize(text)
            return [' '.join(grams) for grams in nltk.ngrams(words, n)]


        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 10))
        df_doc_ngram_10 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        print(os.path.join(output_directory, file['name'] + '_ngram_10'))
        df_doc_ngram_10.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_10.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 9))
        df_doc_ngram_9 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_9.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_9.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 8))
        df_doc_ngram_8 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_8.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_8.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 7))
        df_doc_ngram_7 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_7.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_7.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 6))
        df_doc_ngram_6 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_6.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_6.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 5))
        df_doc_ngram_5 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_5.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_5.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 4))
        df_doc_ngram_4 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_4.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_4.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 3))
        df_doc_ngram_3 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_3.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_3.csv'), index=False)

        df_doc['ngram_text'] = df_doc['text'].apply(lambda x: generate_ngrams(x, 2))
        df_doc_ngram_2 = df_doc.explode('ngram_text').dropna(subset=['ngram_text'])
        df_doc_ngram_2.to_csv(os.path.join(df_doc_output_directory, file['name'] + '_ngram_2.csv'), index=False)

        # reading and pre-processing Sonix Transcripts
        from pysrt import open as open_srt

        def read_srt(file_path):
            subs = open_srt(file_path)
            return [{'n': sub.index, 'start': sub.start.ordinal, 'end': sub.end.ordinal, 'text': sub.text} for sub in subs]


        dfson = pd.DataFrame(read_srt(file['srt']))
        dfson['text'] = dfson['text'].str.replace(r'SPEAKER\d\:|[,.\?!¿…!\[\]"¡]', '', regex=True).str.strip().str.lower()
        dfson['word_n'] = dfson['text'].apply(lambda x: len(re.findall(r'\w+', x)))
        dfson = dfson[dfson['word_n'] >= 5]
        text_counts = dfson['text'].value_counts().reset_index()
        text_counts.columns = ['text', 'text_unique']
        dfson = dfson.merge(text_counts, on='text')
        dfson = dfson[dfson['text_unique'] == 1]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 10))
        dfson_ngram_10 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_10 = dfson_ngram_10[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 9))
        dfson_ngram_9 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_9 = dfson_ngram_9[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 8))
        dfson_ngram_8 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_8 = dfson_ngram_8[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 7))
        dfson_ngram_7 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_7 = dfson_ngram_7[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 6))
        dfson_ngram_6 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_6 = dfson_ngram_6[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 5))
        dfson_ngram_5 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_5 = dfson_ngram_5[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 4))
        dfson_ngram_4 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_4 = dfson_ngram_4[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 3))
        dfson_ngram_3 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_3 = dfson_ngram_3[['n', 'start', 'end', 'ngram_text']]

        dfson['ngram_text'] = dfson['text'].apply(lambda x: generate_ngrams(x, 2))
        dfson_ngram_2 = dfson.explode('ngram_text').dropna(subset=['ngram_text'])
        dfson_ngram_2 = dfson_ngram_2[['n', 'start', 'end', 'ngram_text']]

        # match manual and sonix transcripts
        df_matched_10 = pd.merge(df_doc_ngram_10, dfson_ngram_10, on='ngram_text', how='left')
        df_matched_10 = df_matched_10.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=10)

        df_matched_9 = pd.merge(df_doc_ngram_9, dfson_ngram_9, on='ngram_text', how='left')
        df_matched_9 = df_matched_9.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=9)

        df_matched_8 = pd.merge(df_doc_ngram_8, dfson_ngram_8, on='ngram_text', how='left')
        df_matched_8 = df_matched_8.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=8)

        df_matched_7 = pd.merge(df_doc_ngram_7, dfson_ngram_7, on='ngram_text', how='left')
        df_matched_7 = df_matched_7.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=7)

        df_matched_6 = pd.merge(df_doc_ngram_6, dfson_ngram_6, on='ngram_text', how='left')
        df_matched_6 = df_matched_6.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=6)

        df_matched_5 = pd.merge(df_doc_ngram_5, dfson_ngram_5, on='ngram_text', how='left')
        df_matched_5 = df_matched_5.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=5)

        df_matched_4 = pd.merge(df_doc_ngram_4, dfson_ngram_4, on='ngram_text', how='left')
        df_matched_4 = df_matched_4.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=4)

        df_matched_3 = pd.merge(df_doc_ngram_3, dfson_ngram_3, on='ngram_text', how='left')
        df_matched_3 = df_matched_3.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=3)

        df_matched_2 = pd.merge(df_doc_ngram_2, dfson_ngram_2, on='ngram_text', how='left')
        df_matched_2 = df_matched_2.dropna(subset=['n']).drop_duplicates(subset=['line_number_n']).assign(ngram_number=2)


        for i in range(2, 11):
            print(f'length at ngram {i} df_ngram, dfson, matched',eval(f'df_doc_ngram_{i}').shape,
                  eval(f'dfson_ngram_{i}').shape, eval(f'df_matched_{i}').shape)


        #matching all the dfs
        matches_from_9 = set(df_matched_9['line_number_n']) - set(df_matched_10['line_number_n'])
        df_matched_all = (pd.concat([df_matched_10, df_matched_9[df_matched_9['line_number_n'].isin(matches_from_9)]])
                          .sort_values('line_number_n'))

        matches_from_8 = set(df_matched_8['line_number_n']) - set(df_matched_all['line_number_n'])
        df_matched_all = (pd.concat([df_matched_all, df_matched_8[df_matched_8['line_number_n'].isin(matches_from_8)]])
                          .sort_values('line_number_n'))

        matches_from_7 = set(df_matched_7['line_number_n']) - set(df_matched_all['line_number_n'])
        df_matched_all = (pd.concat([df_matched_all, df_matched_7[df_matched_7['line_number_n'].isin(matches_from_7)]])
                          .sort_values('line_number_n'))

        matches_from_6 = set(df_matched_6['line_number_n']) - set(df_matched_all['line_number_n'])
        df_matched_all = (pd.concat([df_matched_all, df_matched_6[df_matched_6['line_number_n'].isin(matches_from_6)]])
                          .sort_values('line_number_n'))

        matches_from_5 = set(df_matched_5['line_number_n']) - set(df_matched_all['line_number_n'])
        df_matched_all = (pd.concat([df_matched_all, df_matched_5[df_matched_5['line_number_n'].isin(matches_from_5)]])
                          .sort_values('line_number_n'))

        matches_from_4 = set(df_matched_4['line_number_n']) - set(df_matched_all['line_number_n'])
        df_matched_all = (pd.concat([df_matched_all, df_matched_4[df_matched_4['line_number_n'].isin(matches_from_4)]])
                          .sort_values('line_number_n'))

        df_speaker = df_matched_all[df_matched_all['speaker'] == 'Hablante']
        df_speaker = df_speaker[~(df_speaker['end'] < df_speaker['start'])]

        # for _ in range(10):
        #     for rowi in range(len(df_speaker) - 1, 0, -1):
        #         tmp_start = df_speaker.iloc[rowi]['start']
        #         tmp_start_previous = df_speaker.iloc[rowi - 1]['start']
        #         if tmp_start < tmp_start_previous:
        #             df_speaker = df_speaker.drop(df_speaker.index[rowi])

        df_speaker = df_speaker.drop_duplicates(subset=['start'])

        # for rowi in range(len(df_speaker) - 1):
        #     tmp_end = df_speaker.iloc[rowi]['end']
        #     tmp_start_next = df_speaker.iloc[rowi + 1]['start']
        #     if tmp_end > tmp_start_next:
        #         df_speaker.at[df_speaker.index[rowi], 'end'] = tmp_start_next

        tgdur = file['dur']

        tmpspeakerdf = df_speaker[df_speaker['end'] > df_speaker['start']]
        tmpspeakerdf['start'] = tmpspeakerdf['start'].astype(float) / 1000
        tmpspeakerdf['end'] = tmpspeakerdf['end'].astype(float) / 1000
        tmpspeakerdf = tmpspeakerdf[tmpspeakerdf['start'] <= tgdur]
        tmpspeakerdf = tmpspeakerdf[tmpspeakerdf['end'] <= tgdur]
        tmpspeakerdf = tmpspeakerdf.sort_values(by='start')


        # df_speaker_out.to_csv('Output/SpeakingTurns_CA1HA_87.csv', index=False, sep=',', quoting=3)
        # Create a TextGrid using parselmouth
        tg = tgt.TextGrid()

        # Create a new interval tier
        tier = tgt.IntervalTier( 0, tgdur, 'Speaker')


        for index, row in tmpspeakerdf.iterrows():
            interval = tgt.Interval(row['start'], row['end'], row['text'])
            tier.add_interval(interval)
        #
        save_name = os.path.join(output_directory, file['name'] + '.TextGrid')
        # call(tg, "Write to text file", save_name)

        tg.add_tier(tier)
        tgt.io.write_to_file(tg, save_name)

if __name__ == '__main__':
    main()