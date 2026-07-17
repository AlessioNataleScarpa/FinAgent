"""
Graph Construction for ETF Analysis Pipeline.
Constructs and compiles StateGraph from langgraph.graph (StateGraph, START, END).
Connects FORK (ISIN -> parallel info_presentazione, news, info_andamenti_storici)
and JOIN (out_1 and out_tech -> join_presenter -> END).
"""

from langgraph.graph import StateGraph, START, END
from .state import PipelineState
from .info_presentazione import fetch_info_presentazione
from .agent_1 import generate_agent_1_out
from .news import fetch_news
from .info_andamenti_storici import fetch_info_andamenti_storici
from .predict import predict_node
from .agent_2 import generate_agent_2_out
from .join_presenter import join_presenter_node

# Instantiate StateGraph with PipelineState
builder = StateGraph(PipelineState)

# Add nodes to graph
builder.add_node("info_presentazione", fetch_info_presentazione)
builder.add_node("agent_1", generate_agent_1_out)
builder.add_node("news", fetch_news)
builder.add_node("info_andamenti_storici", fetch_info_andamenti_storici)
builder.add_node("predict", predict_node)
builder.add_node("agent_2", generate_agent_2_out)
builder.add_node("join_presenter", join_presenter_node)

# Define FORK from START into parallel branches
builder.add_edge(START, "info_presentazione")
builder.add_edge(START, "news")
builder.add_edge(START, "info_andamenti_storici")

# Sequential flow for branch 1: info_presentazione -> agent_1
builder.add_edge("info_presentazione", "agent_1")

# Sequential flow for branch 3: info_andamenti_storici -> predict
builder.add_edge("info_andamenti_storici", "predict")

# agent_2 waits for both news and predict nodes
builder.add_edge("news", "agent_2")
builder.add_edge("predict", "agent_2")

# JOIN: join_presenter waits for both agent_1 (OUT 1) and agent_2 (OUT TECNICA)
builder.add_edge("agent_1", "join_presenter")
builder.add_edge("agent_2", "join_presenter")

# Final transition: join_presenter -> END
builder.add_edge("join_presenter", END)

# Compile graph
graph = builder.compile()
app = graph
