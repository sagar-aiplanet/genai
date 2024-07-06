import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm.llms import AzureOpenAIModel
from beyondllm import source
import secrets
import os


st.title("Chat with ZML file Patient data file.")



endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
api_version = st.secrets.azure_embeddings_credentials.API_VERSION
deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
API_KEY = st.secrets.azure_embeddings_credentials.API_KEY

uploaded_data_files = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True, label_visibility="visible")

def uploaded_files(uploaded_data_files):
    if not uploaded_data_files:
        return None
    else:
        print("uploaded_data_files",uploaded_data_files)
        save_path = "./patient_data"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filenames = []
        for file in uploaded_data_files:
            file_path = os.path.join(save_path, file.name)
            filenames.append(file_path)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        print("filenames",filenames)
        data = source.fit(filenames, dtype="pdf", chunk_size=1024, chunk_overlap=0)
        return data

embed_model = AzureAIEmbeddings(
    endpoint_url = endpoint_url,
    azure_key = azure_key,
    api_version= api_version,
    deployment_name=deployment_name
)

question = st.text_input("Enter your question")
submit=st.button("Get the data")
if submit:
    data = uploaded_files(uploaded_data_files)
    retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
    llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name="gpt-4-32k" ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})    
    if not uploaded_data_files:
        system_prompt = "you will act like a chat ...."
        pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
        st.write("Please upload a file")
    else:
        
        # option = st.selectbox( 'Please Select the Patient name?', ('Bobby Jackson', 'Leslie Terry','Danny Smith'))
        # question = "what is the Bobby Jackson condition?"

        system_prompt = '''
        You are an expert medical AI chatbot. When a user uploads multiple documents, you should analyze and understand the content to determine the category of the questions related to the documents and answer them accordingly.

        Category one - Document and Report Handling:

        When a user uploads patient documents or reports, whether for an individual or an entire family, the chatbot should analyze the content and provide relevant answers based on the uploaded documents.

        Category two - General Health Suggestions:

        When users ask about their health conditions, the chatbot should offer general health suggestions only.
        The chatbot must avoid giving specific medical advice, diagnoses, or medication recommendations.

        Category three - Company Information:
        If users inquire about Zml, the medical records company, the chatbot should provide detailed information about the company, including its services and benefits.'''
    
        print(question)
        pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
        response = pipeline.call()
        st.write(response)
        

