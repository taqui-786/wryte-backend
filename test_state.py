from app.agent import get_chat_state
from langchain_core.messages import messages_to_dict

state = get_chat_state("786")
print("TYPE", type(state))
if state:
    # state is a NamedTuple, so we can convert it to a tuple or list
    state_list = list(state)
    print("State List Length:", len(state_list))
    # Replace messages in the first element (values)
    values = dict(state_list[0])
    if "messages" in values:
        values["messages"] = messages_to_dict(values["messages"])
    state_list[0] = values
    print("Success! Can serialize.")
