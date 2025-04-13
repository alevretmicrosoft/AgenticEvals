import asyncio
import json
import time
from dotenv import load_dotenv
from semantic_kernel.agents import AzureResponsesAgent
from semantic_kernel.contents import (
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)

# Import your skill plugins
from skills.booking_skill import BookingPlugin
from skills.dining_skill import DiningPlugin
from skills.semantic_search_plugin import SemanticRoomSearchPlugin
from skills.time_skill import TimePlugin

# Load environment variables
load_dotenv()

# Global list to accumulate intermediate messages per query.
intermediate_steps = []
# Dictionary to track tool call timestamps (for duration measurements).
_tool_call_timestamps = {}

async def handle_intermediate_steps(message: ChatMessageContent):
    """
    Callback to capture intermediate messages, including tool calls and results.
    """
    global intermediate_steps
    intermediate_steps.append(message)
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"🛠️  Tool Call → {item.name}")
            print(f"   Arguments → {item.arguments}")
            _tool_call_timestamps[item.name] = time.time()
        elif isinstance(item, FunctionResultContent):
            start_time = _tool_call_timestamps.get(item.name)
            duration = f"{(time.time() - start_time):.2f}s" if start_time else "N/A"
            print(f"✅  Tool Result from {item.name} (duration: {duration})")
            print(f"   → {item.result}")

async def run_simulation():
    """
    Sets up the agent, runs a series of simulated queries against the agent,
    and logs the entire interaction (user query, tool calls, and final response)
    into a JSONL evaluation file.
    """
    # Setup agent resources and instantiate the agent with your skills.
    client, model = AzureResponsesAgent.setup_resources()
    concierge_agent = AzureResponsesAgent(
        ai_model_id=model,
        client=client,
        name="ConciergeAgent",
        instructions="""
        You are the smart concierge of a luxury hotel. Your name is "Lobby Boy".

        Your job is to help users with:
        - Room availability checks and bookings.
        - Dining queries and table reservations.
        - Providing dates and searching for rooms based on semantic descriptions.

        When interacting:
        - For booking, use check_availability first and then confirm_booking.
        - For dining, use reserve_table for table bookings.
        - For room searches, use search_rooms_by_description ensuring the results match the user’s query.
        - Use get_today and get_relative_date for date-related queries.
                
        Be friendly, helpful, and concise.
        """,
        plugins=[
            BookingPlugin(),
            DiningPlugin(),
            SemanticRoomSearchPlugin(),
            TimePlugin(),
        ],
    )

    # Define a set of simulated user queries covering various skills.
    simulated_queries = [
        "I need a deluxe room for tomorrow. Can you check if any are available?",
        "Please book 1 deluxe room for tomorrow.",
        "What's today's date?",
        "I want to reserve a dinner table for 2 at 19:00.",
        "I'm looking for a room with a sea view. Can you search for me?"
    ]

    # Mapping of tool names to their definitions (reflecting your skill functions).
    tool_definitions_map = {
        "check_availability": {
            "name": "check_availability",
            "description": "Check if a room is available on a certain date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": "Type of room."
                    },
                    "date": {
                        "type": "string",
                        "description": "Booking date in YYYY-MM-DD format."
                    }
                }
            }
        },
        "confirm_booking": {
            "name": "confirm_booking",
            "description": "Confirm booking and reduce room count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": "Type of room."
                    },
                    "date": {
                        "type": "string",
                        "description": "Booking date in YYYY-MM-DD format."
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of rooms to book."
                    }
                }
            }
        },
        "reserve_table": {
            "name": "reserve_table",
            "description": "Simulates table reservation at the hotel restaurant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Reservation time (e.g., HH:MM)."
                    },
                    "party_size": {
                        "type": "integer",
                        "description": "Number of people for the reservation."
                    }
                }
            }
        },
        "search_rooms_by_description": {
            "name": "search_rooms_by_description",
            "description": "Search for hotel rooms by semantic meaning based on room description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The description of the type of room the user is looking for."
                    }
                }
            }
        },
        "get_today": {
            "name": "get_today",
            "description": "Returns today's date in YYYY-MM-DD format.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        "get_relative_date": {
            "name": "get_relative_date",
            "description": "Returns a relative date based on offset in days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_offset": {
                        "type": "integer",
                        "description": "Number of days to add to today."
                    }
                }
            }
        }
    }

    # Open evaluation dataset file for writing in JSONL format.
    with open("evaluation_dataset.jsonl", "w") as file:
        thread = None  # Maintains conversation context if needed.
        for query in simulated_queries:
            print(f"\n👤 User: {query}")
            global intermediate_steps
            intermediate_steps = []  # Reset intermediate messages for current query.
            final_response = ""

            # Invoke the agent with the query; intermediate messages (tool calls/results) are captured via the callback.
            async for response in concierge_agent.invoke(
                messages=query,
                thread=thread,
                on_intermediate_message=handle_intermediate_steps,
                stream=False
            ):
                thread = response.thread
                final_response = str(response.content)  # Explicitly convert to string.
                print(f"# ConciergeAgent: {final_response}\n")

            # Extract tool call details from intermediate steps for logging.
            tool_calls = []
            for message in intermediate_steps:
                for item in message.items:
                    if isinstance(item, FunctionCallContent):
                        # Ensure arguments are JSON-serializable (cast to string if not a dict).
                        arguments = item.arguments if isinstance(item.arguments, dict) else str(item.arguments)
                        tool_call_entry = {
                            "type": "tool_call",
                            "tool_call_id": f"call_{int(time.time() * 1000)}",
                            "name": item.name,
                            "arguments": arguments,
                        }
                        tool_calls.append(tool_call_entry)

            # Derive a list of tool definitions for any tools used during the interaction.
            used_tool_names = {call["name"] for call in tool_calls}
            tool_defs = [
                tool_definitions_map[name]
                for name in used_tool_names
                if name in tool_definitions_map
            ]

            # Build the record for this query simulation.
            record = {
                "query": query,
                "tool_calls": tool_calls,
                "tool_definitions": tool_defs,
                "response": final_response,
            }
            file.write(json.dumps(record) + "\n")

        # Optional cleanup: delete thread if your system requires it.
        if thread:
            await thread.delete()

    print("Simulation completed and interactions have been logged to evaluation_dataset.jsonl.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
