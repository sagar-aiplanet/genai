import pytesseract
from pdf2image import convert_from_path
import os
from llama_index.core import SimpleDirectoryReader

# dr_lal_labpath.pdf"
# DOC-20231221-WA0006.pdf"
# lab_report_deepak_chawla.pdf
file_path = r"/Users/hemasagarendluri1996/aiplanet/genai/Usecases/zml-medical-data/patient_data/lab_report_deepak_chawla.pdf"
os.chdir('/Users/hemasagarendluri1996/aiplanet/genai/Usecases/zml-medical-data/patient_data/deepaks_data')

# Path to the PDF file
pdf_path = file_path
# Convert PDF to images
pages = convert_from_path(pdf_path, 300)  # 300 DPI
# Iterate through all the pages and extract text using OCR
with open('extracted_text.txt', 'w') as text_file:
    for page_num, page in enumerate(pages):
        # Use pytesseract to extract text from the image
        page_text = pytesseract.image_to_string(page)
        # Write the text to the file
        text_file.write(f"Page {page_num + 1}:\n{page_text}\n\n")

reader = SimpleDirectoryReader(input_dir=r"/Users/hemasagarendluri1996/aiplanet/genai/Usecases/zml-medical-data/patient_data/deepaks_data/")
documents = reader.load_data()
print(documents)