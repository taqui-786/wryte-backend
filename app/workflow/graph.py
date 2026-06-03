from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.prebuilt import tools_condition
from app.workflow.state import ChatState, UserContext
from app.workflow.node import chat_node, classify_node, extract_and_save_node, recall_node
from app.workflow.tool import tool_node


def build_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    builder = StateGraph(ChatState, context_schema=UserContext)
    # My Nodes 
    builder.add_node("classify",classify_node)
    builder.add_node("recall",recall_node)
    builder.add_node("extract_and_save", extract_and_save_node)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
    
    # And My Edges (Buddies)
    builder.add_edge(START, "classify")
    builder.add_edge("classify","recall")
    builder.add_edge("recall","chat_node")
    builder.add_conditional_edges("chat_node", tools_condition)
    builder.add_edge("tools","chat_node")
    builder.add_edge("chat_node","extract_and_save")
    builder.add_edge("extract_and_save", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
