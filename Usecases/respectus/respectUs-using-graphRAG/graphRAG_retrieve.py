from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from timeit import default_timer as timer
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
import os

dotenv.load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
# how do i know i connected or not to the neo4j
# database connection status
graph.query("MATCH (n)--() WITH n, COUNT(*) AS c RETURN n ORDER BY c DESC LIMIT 80;")
print("")

# OpenAI
dotenv.load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["AZURE_OPENAI_DEPLOYMENT"] = AZURE_OPENAI_DEPLOYMENT



llm = AzureChatOpenAI(
    azure_deployment="gpt-4-32k",
    api_version="2024-02-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def prettifyChain(chain):
    """ Pretty print the chain response, returning the response as well. """
    def prettychain(question:str):
      response = chain({"question": question},return_only_outputs=True,)
      print(textwrap.fill(response['answer']))
    return prettychain
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding")
vector_search_qa = prettifyChain(RetrievalQAWithSourcesChain.from_chain_type(
    llm, 
    chain_type="stuff", 
    retriever=vector_index.as_retriever(),
))
 
question = "Give me the rule and exceptions for the regulation of export of goods and technology which might contribute to Iran’s capability to manufacture Unmanned Aerial Vehicles (UAVs) to natural or legal persons, entities or bodies in Iran or for use in Iran?"
vector_search_qa(question)