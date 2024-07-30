
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from operator import itemgetter
import streamlit as st


AZURE_OPENAI_ENDPOINT = st.secrets.azure_embeddings_credentials.EMBEDDING_ENDPOINT_URL
AZURE_OPENAI_API_KEY = st.secrets.azure_embeddings_credentials.EMBEDDING_AZURE_KEY

api_version = st.secrets.azure_embeddings_credentials.AZURE_API_KEY
deployment_name = st.secrets.azure_embeddings_credentials.AZURE_BASE_URL

os.environ['AZURE_OPENAI_API_KEY'] = api_version
os.environ["AZURE_OPENAI_ENDPOINT"] = deployment_name

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k",
    api_version="2024-02-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


import os


os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    
    azure_deployment="text-embed-marketplace",
    openai_api_version="2024-02-01",
)


dot_format = """ digraph G {

                // Start node
                start [shape=ellipse, label="Start"];

                // First question
                start -> product_question;
                product_question [shape=Mdiamond, label="What is your product?"];

                // Leads to another stream
                product_question -> leads_another_stream [label="Not related"];
                leads_another_stream [shape=box, label="LEADS TO ANOTHER STREAM"];

                // Second question
                product_question -> uav_question [label="Related"];
                uav_question [shape=Mdiamond, label="Goods contribute to Iran’s UAVs?"];

                // Third question
                uav_question -> classification_question [label="Yes"];
                classification_question [shape=Mdiamond, label="How is your product classified?"];

                // Annex I
                classification_question -> annex1 [label="Annex I"];
                annex1 [shape=box, label="Annex I of council regulation 2021/1821"];
                annex1 -> authorization1 [label=""];
                authorization1 [shape=box, label="Subject to an authorization Luxembourg export council law of 27 June 2024"];

                // Annex II
                classification_question -> annex2 [label="Annex II"];
                annex2 [shape=box, label="Annex II of council regulation 2023/1593"];
                annex2 -> iran_question [label=""];

                // Fourth question
                iran_question -> export_iran [label="Yes"];
                iran_question [shape=Mdiamond, label="Exporting to Iran?"];

                // Fifth question
                iran_question -> export_other_country [label="No"];
                export_other_country [shape=Mdiamond, label="Exporting to another country for use in Iran?"];

                // Not Restricted
                export_other_country -> not_restricted [label="No"];
                not_restricted [shape=box, label="Not Restricted EU Regulation 2023/1529 of 20 July 2023 Article-2"];

                // Sixth question
                export_iran -> contract_question [label="Yes"];
                contract_question [shape=Mdiamond, label="Obligations from contract before 26 July 2023?"];

                // Seventh question
                contract_question -> operation_date_question [label="Yes"];
                operation_date_question [shape=Mdiamond, label="Export operated before 27 October 2023?"];

                // Eighth question
                contract_question -> non_military_use [label="No"];
                non_military_use [shape=Mdiamond, label="Export for non-military use?"];

                // Ninth question
                non_military_use -> non_military_end_user [label="Yes"];
                non_military_end_user [shape=Mdiamond, label="Export for non-military end-user?"];

                // Tenth question
                non_military_end_user -> purpose_question [label="Yes"];
                purpose_question [shape=Mdiamond, label="Purpose of goods or technology?"];

                // Specific purposes
                purpose_question -> medical_purpose [label="Medical purpose"];
                medical_purpose [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                purpose_question -> pharmaceutical_purpose [label="Pharmaceutical purposes"];
                pharmaceutical_purpose [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                purpose_question -> humanitarian_purpose [label="Humanitarian purpose"];
                humanitarian_purpose [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                purpose_question -> health_emergency [label="Health emergency"];
                health_emergency [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                purpose_question -> urgent_prevention [label="Urgent prevention or mitigation"];
                urgent_prevention [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                purpose_question -> natural_disasters [label="Response to natural disasters"];
                natural_disasters [shape=box, label="Subject to an Authorization EU Regulation 2023/1529 of 20 July 2023 Article 2"];

                // Prohibited
                purpose_question -> prohibited [label="Any other purpose"];
                prohibited [shape=box, label="Prohibited EU Regulation 2023"];

                // Other endings
                operation_date_question -> not_restricted [label="Yes"];
                non_military_use -> prohibited [label="No"];
                non_military_end_user -> prohibited [label="No"]; } """

st.title("Respectus decision tree generator")


uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
def uploaded_files(uploaded_file):
    if uploaded_file is not None:
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            data = UnstructuredFileLoader(file_path).load()
            # raw_doc = loader.
            return data

few_shot_examples = [
{"input":"Give me the rule and exceptions for the regulation of export of goods and technology which might contribute to Iran’s capability to manufacture Unmanned Aerial Vehicles (UAVs) to natural or legal persons, \
entities or bodies in Iran or for use in Iran?",
"output":dot_format}]

few_shot_template = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=few_shot_template,
    examples=few_shot_examples,
)

negotiate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a business analyst with extensive knowledge of legal documents and regulatory documentation.Your task is to create a use-case model that accurately captures regulatory requirements and compliance obligations from a given context.Begin by thoroughly understanding the rules and exceptions, ensuring no detail is overlooked. Read every article and paragraph meticulously, focusing on key elements such as goods, technology, manufacture, legal entities, and authorization requirements, while also considering any restrictions mentioned. Pay special attention to Annex I and II, especially if they contain multiple paragraphs, to ensure no critical details are missed.Distinguish clearly between prohibitions and obligations to maintain accuracy. Ensure every line is considered, avoiding any omissions.Finally, translate your findings into a detailed digraph G format."),
few_shot_prompt,
    ("user", "{question}"),
    ("user", "{context}")
])


question = st.text_input(label='Type your question')
submit=st.button("Generate results")
if submit:
    question = question
    raw_doc = uploaded_files(uploaded_file)

    # segmenting the document into segments
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(raw_doc)
    # Document Embedding with Chromadb

    docsearch = Chroma.from_documents(texts, embeddings)
    # Connection to query with Chroma indexing using a retriever
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={'k':4}
    )


    # Langchain Expression Language to call our LLM using the prompt template above
    # RAG chain
    negotiate_chain = (
        {"context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")}
        | negotiate_prompt
        | llm
        | StrOutputParser()
        )
    answer = negotiate_chain.invoke({"question":"For what purpose goods and technology are necessary?"})
    import pydot
    dot_content = answer
    # Create a graph from DOT content
    graphs = pydot.graph_from_dot_data(dot_content)
    graph = graphs[0]

    # Save the graph to a PNG file
    graph.write_png('dot_graph_2.png')
    with open('dot_graph_2.png', "rb") as file:
        

        btn = st.download_button(
                label="Download image",
                data=file,
                file_name=image_name,
                mime="image/png"
                )