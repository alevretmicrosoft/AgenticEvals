from typing import Annotated
from semantic_kernel.functions import kernel_function

class DiningPlugin:
    """Plugin to handle dining related queries and table reservations."""

    @kernel_function(description="Provides today's dining specials.")
    def get_specials(self) -> Annotated[str, "Returns the dining specials from the restaurant."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of a specified menu item.")
    def get_item_price(self, menu_item: Annotated[str, "Menu item name"]) -> Annotated[str, "Returns the price for the menu item."]:
        return "$9.99"  # This could be extended with dynamic pricing if needed

    @kernel_function(description="Simulates table reservation at the hotel restaurant.")
    def reserve_table(self,
                      time: Annotated[str, "Reservation time"],
                      party_size: Annotated[int, "Number of people"]) -> Annotated[str, "Returns table reservation confirmation."]:
        return f"Table reserved for {party_size} people at {time}."
