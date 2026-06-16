from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Send
from app.workflow.action import ACTION_REGISTRY
from app.workflow.nodes.classifer import classifier_node
from app.workflow.nodes.planner import planner_node
from app.workflow.nodes.replanner import replanner_node
from app.workflow.nodes.step_complete import step_complete_node
from app.workflow.nodes.step_dispatcher import step_dispatcher_node
from app.workflow.nodes.step_worker import step_worker_node
from app.workflow.state import ChatState, UserContext
from app.workflow.node import (
    chat_node,
    classify_node,
    finalize_research,
    humanize_finalize_node,
    writer_planning_node,
    recall_node,
    remember_node,
    research_answer_node,
    research_node,
    research_topic_node,
    write_content_node,
)
from app.workflow.tool import tool_node


def should_continue(state: ChatState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "remember"


def route_after_tools(state: ChatState) -> str:
    if state.get("research_requested"):
        return "research_topics"
    if state.get("writer_requested"):
        return "writer_planning_node"
    return "chat_node"


def route_writer_flow(state: ChatState) -> str:
    if state["writer_requested"]:
        if state["writer_iteration"] < 2 and state["writer_output"].get("feedback"):
            return "write_content"
    return "remember"


def route_after_executor(state: ChatState) -> str:
    if state.get("research_requested"):
        return "research_topics"
    if state.get("writer_requested"):
        return "writer_planning_node"
    return "chat_node"


def route_after_step(state: ChatState) -> str:
    plan = state.get("plan")
    idx = state.get("current_step_index", 0)
    if plan and idx < len(plan):
        return "step_executor"
    return "remember"


# Fan-outttttt
def route_to_research(state: ChatState) -> list[Send]:
    topics = state.get("research_topics", [])
    if not topics:
        return [Send("finalize_research", state)]  # Skip if nothing man

    return [Send("research", {"topic": topic}) for topic in topics]


def router_after_classifier(state: ChatState) -> str:
    task_type = state.get("task_type", "chat")
    if task_type == "chat":
        return "chat_node"
    elif task_type == "tool":
        return "tools"
    elif task_type == "plan":
        return "planner"
    else:
        return "chat_node"


def router_after_step_complete(state: ChatState) -> str:
    if state.get("needs_replan"):
        return "replanner"
    plan = state.get("plan", [])
    completed = set(state.get("completed_step_ids", []))
    pending = [s for s in plan if s["status"] == "pending"]
    if pending:
        return "step_dispatcher"
    return "remember"


def route_after_dispatcher(state: ChatState) -> list[Send] | str:
    if not state.get("dispatch_ready"):
        if state.get("dispatch_done"):
            return "remember"
        return "step_dispatcher"
    plan = state.get("plan", [])
    ready_ids = state.get("dispatch_ready_step_ids", [])
    sends = []
    for step_id in ready_ids:
        step = next((s for s in plan if s["id"] == step_id), None)
        if not step:
            continue
        step_state = {
            **state,
            "plan": state.get("plan", []),
            "current_executing_ids": state.get("current_executing_ids", []),
        }

        action_spec = ACTION_REGISTRY.get(step["action"])
        if action_spec and action_spec.state_setup:
            step_state.update(action_spec.state_setup(state, step["params"]))

        sends.append(Send("step_worker", {**step_state, "current_step_id": step["id"]}))

    return sends


def build_graph(
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
) -> CompiledStateGraph:
    builder = StateGraph(ChatState, context_schema=UserContext)
    builder.add_node("classify", classify_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("recall", recall_node)
    builder.add_node("remember", remember_node)
    builder.add_node("chat_node", chat_node)
    builder.add_node("tools", tool_node)
    builder.add_node("research", research_node)
    builder.add_node("research_topics", research_topic_node)
    builder.add_node("finalize_research", finalize_research)
    builder.add_node("research_answer", research_answer_node)
    builder.add_node("writer_planning_node", writer_planning_node)
    builder.add_node("write_content", write_content_node)
    builder.add_node("humanize", humanize_finalize_node)
    # Plan & Execute DAG (NEW)
    builder.add_node("planner", planner_node)
    builder.add_node("step_dispatcher", step_dispatcher_node)
    builder.add_node("step_worker", step_worker_node)
    builder.add_node("step_complete", step_complete_node)
    builder.add_node("replanner", replanner_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "recall")
    builder.add_edge("recall", "classifier")

    builder.add_conditional_edges(
        "classifier",
        router_after_classifier,
        {
            "chat_node": "chat_node",
            "tools": "tools",
            "planner": "planner",
        },
    )

    # builder.add_conditional_edges("step_executor", route_after_executor, {
    #     "research_topics": "research_topics",
    #     "planning_node": "planning_node",
    #     "chat_node": "chat_node",
    # })

    builder.add_conditional_edges(
        "chat_node",
        should_continue,
        {
            "tools": "tools",
            "remember": "remember",
        },
    )

    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "chat_node": "chat_node",
            "research_topics": "research_topics",
            "writer_planning_node": "writer_planning_node",
        },
    )

    # plan flow
    builder.add_edge("planner", "step_dispatcher")
    builder.add_conditional_edges(
        "step_dispatcher",
        route_after_dispatcher,
        {
            "step_worker": "step_worker",
            "remember": "remember",
            "step_dispatcher": "step_dispatcher",
        },
    )
    # builder.add_edge("step_worker", "step_complete") No Need now
    builder.add_conditional_edges(
        "step_complete",
        router_after_step_complete,
        {
            "replanner": "replanner",
            "step_dispatcher": "step_dispatcher",
            "remember": "remember",
        },
    )
    builder.add_edge("replanner", "step_dispatcher")

    # My Research Graph -------
    builder.add_conditional_edges("research_topics", route_to_research)
    builder.add_edge("research", "finalize_research")
    builder.add_edge("finalize_research", "research_answer")
    builder.add_edge("research_answer", "step_complete")

    # My Writer Graph -------
    builder.add_edge("writer_planning_node", "write_content")
    builder.add_edge("write_content", "humanize")
    builder.add_conditional_edges(
        "humanize",
        route_writer_flow,
        {
            "write_content": "write_content",
            "remember": "step_complete",
        },
    )



    builder.add_edge("remember", END)

    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
