from typing import Annotated
from semantic_kernel.functions import kernel_function
from utils.cosmosdb_client import CosmosDBClient
from openai import AzureOpenAI
import os

class SemanticRoomSearchPlugin:
    def __init__(self):
        self.db = CosmosDBClient()
        self.openai = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    def embed_query(self, text: str) -> list[float]:
        return self.openai.embeddings.create(
            input=[text],
            model=self.embedding_model
        ).data[0].embedding

    @kernel_function(description="Search for hotel rooms by semantic meaning.")
    def search_rooms_by_description(
        self,
        query: Annotated[str, "The description of the type of room the user is looking for."]
    ) -> Annotated[str, "Returns a short list of rooms matching the request."]:
        embedding = self.embed_query(query)

        sql_query = """
        SELECT TOP 3 r.roomType, r.description, r.price, r.available, r.date
        FROM rooms r
        ORDER BY VectorDistance(r.vectorDescription, @embedding)
        """

        results = self.db.container.query_items(
            query=sql_query,
            parameters=[
                {"name": "@embedding", "value": embedding}
            ],
            enable_cross_partition_query=True
        )

        output = ""
        for item in results:
            output += (
                f"\n🏨 **{item['roomType'].capitalize()}**\n"
                f"📅 Date: {item['date']}\n"
                f"🛏 Description: {item['description']}\n"
                f"💵 Price: {item['price']}\n"
                f"🟢 Available: {item['available']} rooms\n"
            )

        return output or "Sorry, no rooms matched your description."
