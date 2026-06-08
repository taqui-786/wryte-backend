from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Send, Command
from app.workflow.state import ChatState, UserContext
from app.workflow.node import chat_node, classify_node, finalize_research, recall_node, remember_node, research_answer_node, research_node, research_topic_node
from app.workflow.tool import tool_node



def should_continue(state: ChatState) -> str:
    """Decide: do we need tools, or are we done chatting?"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "remember"


def route_after_tools(state: ChatState) -> str:
    if state.get("research_requested"):
        return "research_topics"
    return "chat_node"

# Fan-outttttt
def route_to_research(state:ChatState) -> list[Send]:
    topics = state.get("research_topics",[])
    if not topics:
        return [Send("finalize_research", state)] # Skip if nothing man
    
    return [Send("research", {"topic": topic}) for topic in topics]



def build_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    builder = StateGraph(ChatState, context_schema=UserContext)
    # My Nodes 
    builder.add_node("classify",classify_node)
    builder.add_node("recall",recall_node)
    builder.add_node("remember", remember_node)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
    # My research nodes
    builder.add_node("research", research_node)
    builder.add_node("research_topics", research_topic_node)
    builder.add_node("finalize_research", finalize_research)
    builder.add_node("research_answer", research_answer_node)
    
    
    # And My Edges (Buddies)
    builder.add_edge(START, "classify")
    builder.add_edge("classify","recall")
    builder.add_edge("recall","chat_node")

    builder.add_conditional_edges("chat_node", should_continue,{
        "tools": "tools",
        "remember": "remember",
        "research_topics": "research_topics",
    })
    
    builder.add_conditional_edges("tools", route_after_tools, {
        "chat_node": "chat_node",
        "research_topics": "research_topics",
    })
    builder.add_edge("remember", END)

    # Some cool stuff here
    builder.add_conditional_edges("research_topics", route_to_research)
    builder.add_edge("research","finalize_research")
    builder.add_edge("finalize_research","research_answer")
    builder.add_edge("research_answer","remember")

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
