import streamlit as st
import graphviz

# Define DOT content
dot_content = """
digraph G {
  start [shape=ellipse, label="Start"];
  start -> question1 [label="Next"];
  question1 [shape=diamond, label="Is this a question?"];
  question1 -> end [label="Yes"];
  question1 -> start [label="No"];
  end [shape=ellipse, label="End"];
}
"""
submit=st.button("Generate results")
if submit:
    # Create a Graphviz source object
    graph = graphviz.Source(dot_content)

    st.write("Graph Visualization")
    st.graphviz_chart(dot_content, use_container_width=True)