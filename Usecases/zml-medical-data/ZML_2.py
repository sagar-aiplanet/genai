import pytesseract
from pdf2image import convert_from_path
import os
from llama_index.core import SimpleDirectoryReader

# dr_lal_labpath.pdf"
# DOC-20231221-WA0006.pdf"
# lab_report_deepak_chawla.pdf
file_path = r"/Users/hemasagarendluri1996/aiplanet/genai/unclear_pdf/lab_report_deepak_chawla.pdf"
input_file = "/Users/hemasagarendluri1996/aiplanet/Usecases/zml-medical-data/patient_data/Medical_record.pdf"
# os.chdir('/Users/hemasagarendluri1996/aiplanet/genai/Usecases/zml-medical-data/patient_data/deepaks_data')
# from beyondllm import source

# data = source.fit(input_file = ['./patient_data'], dtype="pdf", chunk_size=1024, chunk_overlap=0)
# print(data)

# # Path to the PDF file
# pdf_path = file_path
# # Convert PDF to images
# pages = convert_from_path(pdf_path, 300)  # 300 DPI
# # Iterate through all the pages and extract text using OCR
# with open('extracted_text.txt', 'w') as text_file:
#     for page_num, page in enumerate(pages):
#         # Use pytesseract to extract text from the image
#         page_text = pytesseract.image_to_string(page)
#         # Write the text to the file
#         text_file.write(f"Page {page_num + 1}:\n{page_text}\n\n")

# reader = SimpleDirectoryReader(input_dir=r"/Users/hemasagarendluri1996/aiplanet/genai/Usecases/zml-medical-data/patient_data/deepaks_data/")
# documents = reader.load_data()
# print(documents)


import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.filter(ImageFilter.MedianFilter())  # Apply a median filter
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast
    img = img.point(lambda x: 0 if x < 140 else 255, '1')  # Apply thresholding
    return img

# Change the current working directory
# os.chdir('/home/asylumax/Desktop')

# Path to the PDF file
pdf_path = file_path

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Create a directory to save the images
if not os.path.exists('pdf_images'):
    os.makedirs('pdf_images')

# Extract text from each page and save to a text file
with open('extracted_text.txt', 'w') as text_file:
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Convert PDF page to image
        pix = page.get_pixmap(dpi=300)  # Set a high DPI
        
        # Save the image
        image_path = f'pdf_images/page_{page_num + 1}.png'
        pix.save(image_path)
        
        # Preprocess the image
        img = preprocess_image(image_path)
        
        # Use pytesseract to extract text from the image
        page_text = pytesseract.image_to_string(img)
        
        # Write the text to the file
        text_file.write(f"Page {page_num + 1}:\n{page_text}\n\n")
        
        # Optionally, print the extracted text
        print(f"Page {page_num + 1}:\n{page_text}\n")

# Optionally, delete the images after processing
import shutil
shutil.rmtree('pdf_images')

