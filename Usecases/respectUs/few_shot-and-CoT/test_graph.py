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

# Create a Graphviz source object
graph = graphviz.Source(dot_content)

# Render the graph to a PNG file and display it in Streamlit
try:
    graph.render(filename='graph', format='png', cleanup=True)
    img_bytes = open('graph.png', 'rb').read()
    
    # Display the image in Streamlit
    st.write("Graph Visualization")
    st.image(img_bytes, use_column_width=True)
    
    # Provide a download button for the PNG image
    st.download_button(
        label="Download Graph as PNG",
        data=img_bytes,
        file_name="graph.png",
        mime="image/png"
    )
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()
