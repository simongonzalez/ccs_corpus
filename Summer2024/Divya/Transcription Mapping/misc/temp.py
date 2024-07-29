from docx import Document
import os

# Path to your .docx file
current_working_directory = os.getcwd()
manual_transcript_file_path = os.path.join(current_working_directory, '..', '..', '01_Microservices',
                                           '1_TranscriptionMapping', 'Input_Output', 'Input', 'CA1HA_87.docx')

# Read and print each line from the .docx file

# Load the .docx file
doc = Document(manual_transcript_file_path)

# Iterate through each paragraph in the document
for paragraph in doc.paragraphs:
    # Split the text of the paragraph into lines
    lines = paragraph.text.split('\n')
    # Iterate through each line in the paragraph
    for line in lines:
        # Print the line
        print(line)


