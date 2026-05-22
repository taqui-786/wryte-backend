from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.prebuilt import tools_condition
from app.workflow.state import ChatState, UserContext
from app.workflow.node import chat_node
from app.workflow.tool import tool_node


def build_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    builder = StateGraph(ChatState, context_schema=UserContext)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "chat_node")
    builder.add_conditional_edges("chat_node", tools_condition)
    builder.add_edge("tools", "chat_node")
    builder.add_edge("chat_node", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
