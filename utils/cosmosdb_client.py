import os
from dotenv import load_dotenv
from openai import AzureOpenAI 
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential

load_dotenv()

credential = DefaultAzureCredential()

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_embeddings(text):
    return client.embeddings.create(
        input = [text], 
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")).data[0].embedding

class CosmosDBClient:
    def __init__(self):
        endpoint = os.getenv("COSMOS_ENDPOINT")
        key = os.getenv("COSMOS_KEY")
        db_name = os.getenv("COSMOS_DB_NAME")
        container_name = os.getenv("COSMOS_CONTAINER_NAME")

        self.client = CosmosClient(endpoint, credential=credential)
        self.database = self.client.create_database_if_not_exists(db_name)
        self.container = self.database.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/roomType"),
            offer_throughput=400
        )

    def get_room_availability(self, room_type: str, date: str):
        query = f"SELECT * FROM rooms r WHERE r.roomType = '{room_type}' AND r.date = '{date}'"
        items = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        return items[0] if items else None

    def update_room_count(self, room_type: str, date: str, count: int):
        doc = self.get_room_availability(room_type, date)
        if doc:
            doc["available"] = max(0, doc["available"] - count)
            self.container.upsert_item(doc)
            return doc
        return None

    def insert_room_record(self, room_type: str, date: str, available: int, price: str, description: str):
        doc_id = f"{room_type}_{date}"
        vector = generate_embeddings(description)

        self.container.upsert_item({
            "id": doc_id,
            "roomType": room_type,
            "date": date,
            "available": available,
            "price": price,
            "description": description,
            "vectorDescription": vector
        })
