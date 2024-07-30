import os
import json
import pydot
import subprocess
import streamlit as st
from getpass import getpass
from beyondllm.llms import AzureOpenAIModel
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm import source,retrieve,embeddings,llms,generator

# endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
# azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
# api_version = st.secrets.azure_embeddings_credentials.API_VERSION
# deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# # DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
# API_KEY = st.secrets.azure_embeddings_credentials.API_KEY

st.title("Respectus decision tree generator")


# api_key = st.text_input("API Key:", type="password")
# os.environ['OPENAI_API_KEY'] = api_key

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

def uploaded_files(uploaded_file):
    if uploaded_file is not None:
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
            return data
data = uploaded_files(uploaded_file)

# embed_model = embeddings.AzureAIEmbeddings(
#                 endpoint_url=endpoint_url,
#                 azure_key=azure_key,
#                 api_version=api_version,
#                 deployment_name=deployment_name)
# BASE_URL = BASE_URL
# DEPLOYMENT_NAME = "gpt-4-32k" 
# API_KEY = API_KEY
embed_model = embeddings.AzureAIEmbeddings(
                    endpoint_url="https://marketplace.openai.azure.com/",
                    azure_key="d6d9522a01c74836907af2f3fd72ff85",
                    api_version="2024-02-01",
                    deployment_name="text-embed-marketplace")
BASE_URL = "https://gpt-res.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-4-32k" 
API_KEY = "a20bc67dbd7c47ed8c978bbcfdacf930"

retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)
llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name=DEPLOYMENT_NAME ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})

question = st.text_input(label='Type your question')
submit=st.button("Generate results")
if submit:
    question = question
    system_prompt = """
                
                Please generate a DOT file format for a decision-making flowchart based on the following steps and conditions.
                





"""


    
    pipeline = generator.Generate(question=question, system_prompt=system_prompt, retriever=retriever, llm=llm)
    decision_tree_json = pipeline.call()
    # print(decision_tree_json)
    # data = json.loads(decision_tree_json)
    # print(data)
    # For user queiry control 

# ----------------------------------------------------------------------------------------------------------------------------------------------------
    def json_to_graph(graph, parent_label, parent_node):
        if isinstance(parent_node, dict):
            for key, value in parent_node.items():
                if key in ["Yes", "No"]:
                    option_label = parent_label + "_" + key
                    node = pydot.Node(option_label, label=key, shape='box', style='filled', fillcolor='lightgreen' if key == "Yes" else 'lightcoral')
                    graph.add_node(node)
                    edge = pydot.Edge(parent_label, option_label)
                    graph.add_edge(edge)
                    
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key.startswith("Question"):
                                question_label = option_label + "_" + sub_key
                                question_text = "\n".join(sub_value[i:i+30] for i in range(0, len(sub_value), 30))
                                node = pydot.Node(question_label, label=question_text, shape='box', style='filled', fillcolor='lightblue')
                                graph.add_node(node)
                                edge = pydot.Edge(option_label, question_label)
                                graph.add_edge(edge)
                                json_to_graph(graph, question_label, value)
                            elif sub_key == "Result":
                                result_label = option_label + "_" + sub_key
                                result_str = f"{sub_key}: {sub_value}"
                                node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
                                graph.add_node(node)
                                edge = pydot.Edge(option_label, result_label)
                                graph.add_edge(edge)
                    else:
                        json_to_graph(graph, option_label, value)
                elif key == "Result":
                    result_label = parent_label + "_" + key
                    result_str = f"{key}: {value}"
                    node = pydot.Node(result_label, label=result_str, shape='box', style='filled', fillcolor='lightgrey')
                    graph.add_node(node)
                    edge = pydot.Edge(parent_label, result_label)
                    graph.add_edge(edge)
#     graph = pydot.Dot(graph_type='graph')

#     start_label = data['Question1']["Question"]
#     print(start_label)
#     start_question = data["Question1"]
#     start_node = pydot.Node(start_label, label=start_label, shape='box', style='filled', fillcolor='lightblue')
#     graph.add_node(start_node)

#     json_to_graph(graph, start_label, start_question)
#     image_name = "decision_tree.png"
#     graph.write_png(image_name)
# # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Display the graph
    # from PIL import Image
    # img = Image.open(uploaded_file.name+'_tree.png')
    # img.show()

    # with open(image_name, "rb") as file:
    #     btn = st.download_button(
    #             label="Download image",
    #             data=file,
    #             file_name=image_name,
    #             mime="image/png"
    #             )
    with st.chat_message(""):
        st.write(decision_tree_json)


# {
#   "Flow": 
#     {
#       "Question": "What is your product?",
#       {"EndFlowIf": "Product is unrelated to the subsequent questions"
#       "Others": "Leads to another stream"},

#       {
#         "Contribution": "Are your goods and technology contributing to Iran's capability to manufacture UAVs?",
#         {
#             "Question": "How is your product classified?",{
#                 "ProductIf": "Product is falls under Annex I of council regulation 2021/1821",{
#                     "End_the_flow_if": "subject to authorization by the Luxembourg export council law of 27 June 2024, Article NN EU"},

#                 }
#                 "ProceedIf": "Product is falls under Annex II of council regulation 2023/1593",
#                 {"Question": "Are you exporting to a natural or legal person, entity, or body in Iran?",{
#                      "Yes":"is the export based on obligations arising from a contract concluded before 26th July 2023 or an ancillary contract necessary for the execution of a contract concluded before 26 July 2023",

#                 }
               
#                 "No":"are you exporting to another country for use in Iran?",{
#                      "No" : "Not Restricted under EU Regulation 2023/1529 of 20 July 2023, Article 2",
#                     {"Yes":"is the export based on obligations arising from a contract concluded before 26th July 2023 or an ancillary contract necessary for the execution of a contract concluded before 26 July 2023",
#                     {

#                         "Yes": "Is the export operated before 27 October 2023?"
#                         {
#                             "Yes": "Not restricted under EU Regulation 2023/1529 of 20 July 2023, Article 2",
#                             "No": "Is the export for non-military use?"
#                         }
#                         "No": "Is the export for non-military use?"
#                         {
#                             "Yes": "Is the export for a non-military end-user?",

#                             {
#                                 "Yes": "For what purpose are the goods or technology necessary?",
#                                 {
#                                     "Purpose": ["Medical", "pharmaceutical", "humanitarian", "health emergency","urgent prevention or mitigation of an event likely to have a serious and significant impact on human health and safety or on the environment", "response to natural disasters","Other"]
#                                     "Athorization":["Subject Authorization Subject TO AN Authorization EU Regulation 2023/1529 of 20 July 2023, Article 2.", 
#                                     "Subject Authorization Subject TO AN Authorization EU Regulation 2023/1529 of 20 July 2023, Article 2.",
#                                     "Subject Authorization Subject TO AN Authorization EU Regulation 2023/1529 of 20 July 2023, Article 2.",
#                                     "Subject Authorization Subject TO AN Authorization EU Regulation 2023/1529 of 20 July 2023, Article 2.",
#                                     "Subject Authorization Subject TO AN Authorization EU Regulation 2023/1529 of 20 July 2023, Article 2.",
#                                     "Prohibited RU Regulation 2023"]
#                                 }
#                                 "No":"Prohibited under EU Regulation 2023/1529 of 20 July 2023, Article 2
#                             }
#                             "No" : "Prohibited under EU Regulation 2023/1529 of 20 July 2023, Article 2"
#                         }
#                     }
#                     "No" : "Not Restricted under EU Regulation 2023/1529 of 20 July 2023, Article 2",

#                 }
#                 }
#         }
#       }
#     }
# }


    