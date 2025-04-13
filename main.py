import asyncio
import time
from dotenv import load_dotenv
from semantic_kernel.agents import AzureResponsesAgent
from semantic_kernel.contents import (
    ChatMessageContent, 
    FunctionCallContent, 
    FunctionResultContent,
)

from skills.booking_skill import BookingPlugin
from skills.dining_skill import DiningPlugin
from skills.time_skill import TimePlugin
from skills.semantic_search_plugin import SemanticRoomSearchPlugin

load_dotenv()

intermediate_steps = []
_tool_call_timestamps = {}

async def handle_intermediate_steps(message: ChatMessageContent):
    intermediate_steps.append(message)

    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"\033[1;36m🛠️  Tool Call → {item.name}\033[0m")
            print(f"   Arguments → {item.arguments}")
            _tool_call_timestamps[item.name] = time.time()

        elif isinstance(item, FunctionResultContent):
            start_time = _tool_call_timestamps.get(item.name, None)
            duration = f"{(time.time() - start_time):.2f}s" if start_time else "N/A"
            print(f"\033[1;32m✅ Tool Result from {item.name} (duration: {duration})\033[0m")
            print(f"   → {item.result}")

async def main():
    client, model = AzureResponsesAgent.setup_resources()

    concierge_agent = AzureResponsesAgent(
        ai_model_id=model,
        client=client,
        name="ConciergeAgent",
        instructions="""
        You are the smart concierge of a luxury hotel. Your name is "Lobby Boy".

        Your job is to help users:
        - Book rooms
        - Check availability
        - Find rooms matching descriptions (e.g. romantic, eco-friendly, workspace)

        When using tools like semantic room search, make sure to:
        - Only return rooms that are truly relevant to the user’s request
        - Do not show rooms that only loosely match (e.g. avoid showing a 'single room' for 'eco-friendly' queries)
        - Prioritize rooms with the highest similarity or that match keywords directly
        - Always explain why you're suggesting a room, if needed

        When using tools like booking, make sure to check availability first and then confirm the booking only if rooms are available (never do both in same step). 
        You have to recap the booking details (e.g., room type, price per night, dates, total amount) before confirming.

        Be friendly, helpful, and concise in your responses.
        
        You speak many languages, but your default language is English.
        If the user speaks another language, you can switch to that language.
        """,
        plugins=[
            BookingPlugin(),
            DiningPlugin(),
            SemanticRoomSearchPlugin(),
            TimePlugin(),
        ],
    )

    thread = None

    print("🛎️  Welcome to the Smart Hospitality Assistant")
    print("Type your message below. Type 'exit' to quit.\n")

    while True:
        user_input = input("👤 User: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        async for response in concierge_agent.invoke(
            messages=user_input,
            thread=thread,
            on_intermediate_message=handle_intermediate_steps,
            stream=False,
        ):
            thread = response.thread
            print(f"# ConciergeAgent: {response.content}\n")

    await thread.delete() if thread else None
    print("👋 Session ended.")

if __name__ == "__main__":
    asyncio.run(main())
