import fitz  # PyMuPDF
import os
import pytesseract
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from PIL import Image, ImageEnhance, ImageFilter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())

# loading embedding keys
EMBEDDING_ENDPOINT_URL = os.environ.get("EMBEDDING_ENDPOINT_URL")
EMBEDDING_AZURE_KEY = os.environ.get("EMBEDDING_AZURE_KEY")
# loading chatopenai model keys
AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")

os.environ['AZURE_OPENAI_API_KEY'] = AZURE_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_BASE_URL

llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k",
    api_version="2024-02-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


os.environ["AZURE_OPENAI_API_KEY"] = EMBEDDING_AZURE_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = EMBEDDING_ENDPOINT_URL


embeddings = AzureOpenAIEmbeddings(
    
    azure_deployment="text-embed-marketplace",
    openai_api_version="2024-02-01",
)

uploaded_data_files = st.file_uploader("Upload files", type=["pdf","txt"], accept_multiple_files=True, label_visibility="visible")

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.filter(ImageFilter.MedianFilter())  # Apply a median filter
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # Increase contrast
    img = img.point(lambda x: 0 if x < 140 else 255, '1')  # Apply thresholding
    return img

def tesseract(filenames):

    if not os.path.exists('pdf_images'):
        os.makedirs('pdf_images')
    for file_path in filenames:
        # Replace with text file extension
        pdf_document = fitz.open(file_path)
        text_file_path = file_path[:-3]+'txt'
        # Extract text from each page and save to a text file
        with open(text_file_path, 'w') as text_file:
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300)  #convert pdf to image
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
                # print(f"Page {page_num + 1}:\n{page_text}\n")
        # Optionally, delete the images after processing
    # import shutil
    # shutil.rmtree('pdf_images')
    
def uploaded_files(uploaded_data_files):
    import shutil
    if not uploaded_data_files:
        return None
    else:
        print("uploaded_data_files",uploaded_data_files)
        save_path = "./datafiles"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filenames = []
        for file in uploaded_data_files:
            file_path = os.path.join(save_path, file.name)
            filenames.append(file_path)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        tesseract(filenames)
        # reader = SimpleDirectoryReader(input_dir="./datafiles")
        # documents = reader.load_data()
        from langchain_community.document_loaders import DirectoryLoader
        loader = DirectoryLoader("./datafiles")
        docs = loader.load()
        print("documents",docs)
        shutil.rmtree('datafiles')
        shutil.rmtree('pdf_images')
        return docs

question = st.text_input("Enter your question")
submit=st.button("Get the data")
if submit:
    data = uploaded_files(uploaded_data_files)

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    text = text_splitter.split_documents(data)

    docsearch  =  Chroma.from_documents(text,embeddings)

    retriever = docsearch.as_retriever(
        search_type = "similarity",
        search_kwags = {'k': 4}

    )
    # docs = retriever.get_relevant_documents("what is the patient name?")
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Build prompt
    template = """
    You are an expert medical AI chatbot. When a user uploads multiple documents, \
    you should analyze and understand the content to determine the category of the questions related to the documents and answer them accordingly.

    Extract without rephrasing all medical conditions, diagnosis, medical problem, medical symptom entities from the context.
    Extract without rephrasing all medical treatment, medical procedure, medical intervention, medication, drug entities from the context.
    Extract without rephrasing all vital signs, laboratory test, medical test, imaging study, diagnostic test entities from the context.

    More you should focus on five things:

    Analyze historical patient data to identify patterns and risk factors that can improve diagnosis, treatment, and prevention strategies. \
    Provide data-driven insights and recommendations based on the patient's medical history and similar case studies to support clinical decision-making. \
    Enable early detection of potential health issues by identifying warning signs from past data, and analyze trends in patient data over time to detect significant changes or developments in health conditions.


    {context}

    Question: {question}

    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    # question = "Is probability a class topic?"
    result = qa_chain({"query": question})
    # Check the result of the query
    # result["result"]
    # # Check the source document from where we 
    # result["source_documents"][0]
    st.write(result["result"])


  