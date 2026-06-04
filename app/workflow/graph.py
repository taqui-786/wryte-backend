from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.prebuilt import tools_condition
from app.workflow.state import ChatState, UserContext
from app.workflow.node import chat_node, classify_node, recall_node, remember_node
from app.workflow.tool import tool_node



def should_continue(state: ChatState) -> str:
    """Decide: do we need tools, or are we done chatting?"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "remember"  # <-- not END! remember comes next.

def after_tools(state: ChatState) -> str:
    """After tools run, always go back to chat (the LLM needs the tool result)."""
    return "chat"

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
    
    # And My Edges (Buddies)
    builder.add_edge(START, "classify")
    builder.add_edge("classify","recall")
    builder.add_edge("recall","chat_node")

    builder.add_conditional_edges("chat_node", should_continue,{
        "tools": "tools",
        "remember": "remember",
    })
    
    builder.add_edge("tools","chat_node")
    builder.add_edge("remember", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
