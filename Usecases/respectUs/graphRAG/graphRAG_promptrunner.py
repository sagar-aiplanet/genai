import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
import dotenv
import os

dotenv.load_dotenv()
endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY

os.environ['AZURE_OPENAI_API_KEY'] = azure_key
os.environ["AZURE_OPENAI_ENDPOINT"] =endpoint_url




from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k",
    api_version="2024-02-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


CYPHER_QA_TEMPLATE = """
  You have comprehensive knowledge of EU legal documents, regulations, and how they interact with various articles and annexes.\
  You provide accurate, concise, and detailed explanations of specific Council Regulations.
  create like a question and answers. return decision tree. 
  

Information:

{context}

Question: {question}

Do not add any conclusion.
"""


qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

def query_graph(user_input):
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        qa_prompt=qa_prompt
        )
    result = chain(user_input)
    return result
# user_input = "Give me the rule and exceptions for the regulation of export of goods and technology which might contribute to Iran’s capability to manufacture Unmanned Aerial Vehicles (UAVs) to natural or legal persons, entities or bodies in Iran or for use in Iran?"

user_input = st.text_input("Enter your question", key="input")
submit=st.button("Generate results")
if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            result = query_graph(user_input)
            answer = result["result"]
            st.write(answer)

        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)
    st.write(f"Time taken: {timer() - start:.2f}s")