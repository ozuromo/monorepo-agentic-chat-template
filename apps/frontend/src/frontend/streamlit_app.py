# type: ignore
import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage

from frontend.client import AgentClient, AgentClientError

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.

APP_TITLE = "Chat Inteligente"
APP_ICON = "💬"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """Get the user ID from session state or URL parameters, or create a new one if it doesn't exist."""
    # Check if user_id exists in session state
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # Try to get from URL parameters using the new st.query_params
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # Generate a new user_id if not found
    user_id = str(uuid.uuid4())

    # Store in session state for this session
    st.session_state[USER_ID_COOKIE] = user_id

    # Also add to URL parameters so it can be bookmarked/shared
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # Get or create user ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            try:
                messages: list[AnyMessage] = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # Config options
    with st.sidebar:
        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
            )
            # Display user ID (for debugging or user information)
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [
                    session.client.request.protocol,
                    session.client.request.host,
                    "",
                    "",
                    "",
                    "",
                ]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            # Include both thread_id and user_id in the URL for sharing to maintain user identity
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}&{USER_ID_COOKIE}={user_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

    # Draw existing messages
    messages: list[AnyMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "react_agent":
                WELCOME = "Olá! Eu sou um agente React. Pergunte-me qualquer coisa!"
            case _:
                WELCOME = "Olá! Eu sou um agente. Pergunte-me qualquer coisa!"

        with st.chat_message("ai"):
            st.text(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[AnyMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        st.chat_message("human").text(user_input)
        try:
            output = await agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
                user_id=user_id,
            )

            # Get existing message IDs to avoid duplicates
            existing_ids = {msg.id for msg in messages}

            # Add only new messages that don't already exist
            for msg in output.messages:
                if msg.id not in existing_ids:
                    messages.append(msg)

            st.rerun()  # Rerun to update the chat with new messages
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()


async def draw_messages(
    messages_agen: AsyncGenerator[AnyMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        if not isinstance(msg, BaseMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.text(msg)
            st.stop()

        # A message from the user, the easiest case
        if isinstance(msg, HumanMessage):
            last_message_type = "human"
            # weird bug with newline characters in streamlit
            st.chat_message("human").text(msg.content)

        # A message from the agent is the most complex case, since we need to
        # handle streaming tokens and tool calls.
        elif isinstance(msg, AIMessage):
            # If we're rendering new messages, store the message in session state
            if is_new:
                st.session_state.messages.append(msg)

            # If the last message type was not AI, create a new chat message
            if last_message_type != "ai":
                last_message_type = "ai"
                st.session_state.last_message = st.chat_message("ai")

            with st.session_state.last_message:
                # If the message has content, write it out.
                # Reset the streaming variables to prepare for the next message.
                if msg.content:
                    # when using write streamlit uses markdown, so we need to fix the newlines
                    st.write(msg.content.replace("\n", "  \n"))

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Create a status container for each tool call and store the
                    # status container by ID to ensure results are mapped to the
                    # correct status container.
                    call_results = {}
                    for tool_call in msg.tool_calls:
                        status = st.status(
                            f"""Tool Call: {tool_call["name"]}""",
                            state="running" if is_new else "complete",
                        )
                        call_results[tool_call["id"]] = status
                        status.write("Input:")
                        status.write(tool_call["args"])

                    # Expect one ToolMessage for each tool call.
                    for tool_call in msg.tool_calls:
                        if "transfer_to" in tool_call["name"]:
                            await handle_agent_msgs(messages_agen, call_results, is_new)
                            break
                        tool_result: AnyMessage = await anext(messages_agen)

                        if tool_result.type != "tool":
                            st.error(f"Unexpected AnyMessage type: {tool_result.type}")
                            st.text(tool_result)
                            st.stop()

                        # Record the message if it's new, and update the correct
                        # status container with the result
                        if is_new:
                            st.session_state.messages.append(tool_result)
                        if tool_result.tool_call_id:
                            status = call_results[tool_result.tool_call_id]
                        status.write("Output:")
                        status.text(tool_result.content)
                        status.update(state="complete")

        # In case of an unexpected message type, log an error and stop
        else:
            st.error(f"Unexpected AnyMessage type: {msg.type}")
            st.text(msg)
            st.stop()


async def handle_agent_msgs(messages_agen, call_results, is_new):
    """
    This function segregates agent output into a status container.
    It handles all messages after the initial tool call message
    until it reaches the final AI message.
    """
    nested_popovers = {}
    # looking for the Success tool call message
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)
    status = call_results.get(getattr(first_msg, "tool_call_id", None))
    # Process first message
    if status and first_msg.content:
        status.text(first_msg.content)
        # Continue reading until finish_reason='stop'
    while True:
        # Check for completion on current message
        finish_reason = getattr(first_msg, "response_metadata", {}).get("finish_reason")
        # Break out of status container if finish_reason is anything other than "tool_calls"
        if finish_reason is not None and finish_reason != "tool_calls":
            if status:
                status.update(state="complete")
            break
        # Read next message
        sub_msg = await anext(messages_agen)
        # this should only happen is skip_stream flag is removed
        # if isinstance(sub_msg, str):
        #     continue
        if is_new:
            st.session_state.messages.append(sub_msg)

        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.text(sub_msg.content)
            first_msg = sub_msg
            continue
        # Display content and tool calls using the same status
        if status:
            if sub_msg.content:
                status.text(sub_msg.content)
            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    popover = status.popover(f"{tc['name']}", icon="🛠️")
                    popover.write(f"**Tool:** {tc['name']}")
                    popover.write("**Input:**")
                    popover.write(tc["args"])
                    # Store the popover reference using the tool call ID
                    nested_popovers[tc["id"]] = popover
        # Update first_msg for next iteration
        first_msg = sub_msg


if __name__ == "__main__":
    asyncio.run(main())
