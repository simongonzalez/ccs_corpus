from docx import Document
from subprocess import call
import os
import pypandoc

def convert_doc_to_docx(input_path, output_path):
    pypandoc.convert_file(input_path, 'docx', outputfile=output_path)

current_working_directory = os.getcwd()
manual_transcript_file_path = os.path.join(current_working_directory, '..', '..', '01_Microservices',
                                           '1_TranscriptionMapping', 'Input_Output', 'Input', 'CA1HA_87.doc')
input_path = manual_transcript_file_path
output_path = os.path.dirname(manual_transcript_file_path)

call(['libreoffice', '--convert-to', 'docx', input_path, '--outdir', output_path])
print(f"File converted and saved to {output_path}")
