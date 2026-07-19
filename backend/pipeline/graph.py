"""
Graph Construction for ETF Analysis Pipeline.

Flow:
  START
    └─ route_mode
         ├─ conversation ──────────────────────────────────────────────→ END
         └─ start_analysis
              ├─ info_presentazione → agent_1 → composition_charts ────┐
              ├─ news ──────────────────────────┐                      │
              └─ info_andamenti_storici ┬→ predict ┴→ agent_2 → sentiment_charts
                                        └→ timeline_charts ────────────┼→ join_presenter → save_memory → END
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .agent_1 import generate_agent_1_out
from .agent_2 import generate_agent_2_out
from .composition_charts import composition_charts_node
from .conversation import conversation_node
from .info_andamenti_storici import fetch_info_andamenti_storici
from .info_presentazione import fetch_info_presentazione
from .join_presenter import join_presenter_node
from .news import fetch_news
from .predict import predict_node
from .save_memory import save_memory_node
from .sentiment_charts import sentiment_charts_node
from .state import PipelineState
from .timeline_charts import timeline_charts_node


def route_mode(state: PipelineState) -> Literal["conversation", "full_analysis"]:
    mode = state.get("mode") or "full_analysis"
    if mode == "conversation":
        return "conversation"
    return "full_analysis"


def start_analysis_node(state: PipelineState) -> dict:
    """Passthrough hub that fans out into the parallel analysis branches."""
    return {}


builder = StateGraph(PipelineState)

# Entry routing
builder.add_node("start_analysis", start_analysis_node)
builder.add_node("conversation", conversation_node)

# Full analysis nodes
builder.add_node("info_presentazione", fetch_info_presentazione)
builder.add_node("agent_1", generate_agent_1_out)
builder.add_node("composition_charts", composition_charts_node)
builder.add_node("news", fetch_news)
builder.add_node("info_andamenti_storici", fetch_info_andamenti_storici)
builder.add_node("predict", predict_node)
builder.add_node("timeline_charts", timeline_charts_node)
builder.add_node("agent_2", generate_agent_2_out)
builder.add_node("sentiment_charts", sentiment_charts_node)
builder.add_node("join_presenter", join_presenter_node)
builder.add_node("save_memory", save_memory_node)

# Conditional entry: conversation vs full pipeline
builder.add_conditional_edges(
    START,
    route_mode,
    {
        "conversation": "conversation",
        "full_analysis": "start_analysis",
    },
)
builder.add_edge("conversation", END)

# Parallel data collection
builder.add_edge("start_analysis", "info_presentazione")
builder.add_edge("start_analysis", "news")
builder.add_edge("start_analysis", "info_andamenti_storici")

# Presentation branch → composition pies
builder.add_edge("info_presentazione", "agent_1")
builder.add_edge("agent_1", "composition_charts")

# Historical branch → predict + timeline chart
builder.add_edge("info_andamenti_storici", "predict")
builder.add_edge("info_andamenti_storici", "timeline_charts")

# Technical branch
builder.add_edge("news", "agent_2")
builder.add_edge("predict", "agent_2")
builder.add_edge("agent_2", "sentiment_charts")

# Join waits for all chart/analysis modules
builder.add_edge("composition_charts", "join_presenter")
builder.add_edge("timeline_charts", "join_presenter")
builder.add_edge("sentiment_charts", "join_presenter")

# Persist memory then end
builder.add_edge("join_presenter", "save_memory")
builder.add_edge("save_memory", END)

graph = builder.compile()
app = graph
