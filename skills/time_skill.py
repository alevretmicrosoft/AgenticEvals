from datetime import datetime, timedelta
from semantic_kernel.functions import kernel_function
from typing import Annotated

class TimePlugin:
    """Utility skill to fetch current or relative dates."""

    @kernel_function(description="Returns today's date in YYYY-MM-DD format.")
    def get_today(self) -> Annotated[str, "The current date formatted as YYYY-MM-DD."]:
        return datetime.now().strftime("%Y-%m-%d")

    @kernel_function(description="Returns a relative date based on offset in days.")
    def get_relative_date(self, days_offset: Annotated[int, "Number of days to add to today."]) -> Annotated[str, "A date offset from today in YYYY-MM-DD format."]:
        return (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
