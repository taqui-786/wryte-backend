from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Send, Command
from app.workflow.state import ChatState, UserContext
from app.workflow.node import chat_node, classify_node, finalize_research, humanize_finalize_node, planning_node, recall_node, remember_node, research_answer_node, research_node, research_topic_node, write_content_node
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
    if state.get("writer_requested"):
        return "planning_node"
    return "chat_node"

def route_writer_flow(state: ChatState) -> str:
    if state["writer_requested"]:
        if state["writer_iteration"] < 2 and state["writer_output"].get("feedback"):
            return "write_content"   # correction loop
    return "remember"                # done → go to END


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
    # My writer nodes
    builder.add_node("planning_node", planning_node)
    builder.add_node("write_content", write_content_node)
    builder.add_node("humanize", humanize_finalize_node)
    
    
    # And My Edges (Buddies)
    builder.add_edge(START, "classify")
    builder.add_edge("classify","recall")
    builder.add_edge("recall","chat_node")

    builder.add_conditional_edges("chat_node", should_continue,{
        "tools": "tools",
        "remember": "remember",
    })
    
    builder.add_conditional_edges("tools", route_after_tools, {
        "chat_node": "chat_node",
        "research_topics": "research_topics",
        "planning_node": "planning_node",
    })
    builder.add_edge("remember", END)

    # Some cool stuff here
    builder.add_conditional_edges("research_topics", route_to_research)
    builder.add_edge("research","finalize_research")
    builder.add_edge("finalize_research","research_answer")
    builder.add_edge("research_answer","remember")
    
    # Writer flow
    builder.add_edge("planning_node", "write_content")
    builder.add_edge("write_content", "humanize")
    builder.add_conditional_edges("humanize", route_writer_flow,{
        "write_content":"write_content",
        "remember":"remember",
    })
    
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
