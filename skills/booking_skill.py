from typing import Annotated
from semantic_kernel.functions import kernel_function
from utils.cosmosdb_client import CosmosDBClient

class BookingPlugin:
    def __init__(self):
        self.db = CosmosDBClient()

    @kernel_function(description="Check if a room is available on a certain date.")
    def check_availability(
        self,
        room_type: Annotated[str, "Type of room"],
        date: Annotated[str, "Booking date"]
    ) -> Annotated[str, "Availability info"]:
        room_type = room_type.lower()
        room = self.db.get_room_availability(room_type, date)
        if room and room["available"] > 0:
            return f"{room['available']} {room_type} rooms available on {date}. Price: {room['price']}"
        return f"Sorry, no {room_type} rooms available on {date}."

    @kernel_function(description="Confirm booking and reduce room count.")
    def confirm_booking(
        self,
        room_type: Annotated[str, "Type of room"],
        date: Annotated[str, "Booking date"],
        count: Annotated[int, "How many rooms to book"]
    ) -> Annotated[str, "Confirmation message"]:
        room_type = room_type.lower()
        room = self.db.get_room_availability(room_type, date)
        if not room:
            return f"No {room_type} rooms found for {date}."
        if room["available"] < count:
            return f"Only {room['available']} {room_type} rooms available for {date}."
        updated = self.db.update_room_count(room_type, date, count)
        return f"✅ Booking confirmed for {count} {room_type} room(s) on {date} at {room['price']}."
