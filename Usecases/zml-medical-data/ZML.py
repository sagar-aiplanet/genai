import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm.llms import AzureOpenAIModel
from beyondllm import source
import secrets
import os
st.title("Chat with ZML file Patient data file.")



embed_model = embeddings.AzureAIEmbeddings(
                    endpoint_url="https://marketplace.openai.azure.com/",
                    azure_key="d6d9522a01c74836907af2f3fd72ff85",
                    api_version="2024-02-01",
                    deployment_name="text-embed-marketplace")
BASE_URL = "https://gpt-res.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4-32k" 
API_KEY = "a20bc67dbd7c47ed8c978bbcfdacf930"
# endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
# azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
# api_version = st.secrets.azure_embeddings_credentials.API_VERSION
# deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# # DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# API_KEY = st.secrets.azure_embeddings_credentials.API_KEY

uploaded_data_files = st.file_uploader("Upload files", type=["pdf","txt"], accept_multiple_files=True, label_visibility="visible")

def uploaded_files(uploaded_data_files):
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
            print("filenames",filenames)
        data = source.fit(filenames, dtype="pdf", chunk_size=1024, chunk_overlap=0)
        # reader = SimpleDirectoryReader(input_dir=filenames[0])
        # documents = reader.load_data()
        return data

        

# embed_model = AzureAIEmbeddings(
#     endpoint_url = endpoint_url,
#     azure_key = azure_key,
#     api_version= api_version,
#     deployment_name=deployment_name
# )

question = st.text_input("Enter your question")
submit=st.button("Get the data")
if submit:
    data = uploaded_files(uploaded_data_files)
    print("data_which we got",data)
    retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
    llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name="gpt-4-32k" ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})    
    if not uploaded_data_files:
        # system_prompt = "You are an AI assistant...."
        # pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
        # response = pipeline.call()
        st.write("Hi,Please upload a pdf file of Patient")
    else:
        
        # option = st.selectbox( 'Please Select the Patient name?', ('Bobby Jackson', 'Leslie Terry','Danny Smith'))
        # question = "what is the Bobby Jackson condition?"

        system_prompt = '''
        

        Context: {context}
        \ You are an expert medical AI chatbot. When a user uploads multiple documents, you should analyze and understand the content to determine the category of the questions related to the documents and answer them accordingly.
        Start with Name: 
        Category one - Document and Report Handling: \ 

        \ When a user uploads patient documents or reports, whether for an individual or an entire family, the chatbot should analyze the content and provide relevant answers based on the uploaded documents.

        Category two - General Health Suggestions: \

        When users ask about their health conditions, the chatbot should offer general health suggestions only.\
        The chatbot must avoid giving specific medical advice, diagnoses, or medication recommendations. \ 

        Category three - Company Information: \
        If users inquire about Zml, the medical records company, the chatbot should provide detailed information about the company, including its services and benefits.
        You are honest, coherent and don't halluicnate. if you did not find relavent context in the document, you could answer below as mentioned.
        
        if you have medical diagnostic report from laboratory analysis it too. 

        Do NOT use any external resource, hyperlink or reference to answer that is not listed.

        if user ask a question related to patient labs results. you should be answer able to answer from lab any document.

        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.please think rationally and answer from your own knowledge base
        If the context is not relevant, please dont answer the question by using your own knowledge about the topic


        if user asks you like frendly questions.
        start with greetings
        tell about you are a Medical AI research assistant

        Answer:
        '''
    
        print(question)
        pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
        response = pipeline.call()
        st.write(response)
        

