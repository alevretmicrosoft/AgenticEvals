import os
import random
import json
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_embeddings(text):
    """Generate embeddings for the given text using Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=[text],
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
    )
    return response.data[0].embedding

# Initialize Cosmos DB client
cosmos_client = CosmosClient(os.getenv("COSMOS_ENDPOINT"), os.getenv("COSMOS_KEY"))

# Database and container names
database_name = os.getenv("COSMOS_DB_NAME")
container_name = os.getenv("COSMOS_CONTAINER_NAME")
partition_key_path = "/roomType"
vector_dimensions = 1536  # text-3-embedding-small output dimensions

# Create (or get) the database
database = cosmos_client.create_database_if_not_exists(database_name)

# Define the vector embedding policy
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/vectorDescription",
            "dataType": "float32",
            "dimensions": vector_dimensions,
            "distanceFunction": "cosine"
        }
    ]
}

# Define the indexing policy
indexing_policy = {
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/\"_etag\"/?",
            "path": "/vectorDescription/*"
        }
    ],
    "vectorIndexes": [
        {
            "path": "/vectorDescription",
            "type": "quantizedFlat"
        }
    ]
}


# Create (or get) the container with the specified policies
try:
    container = database.create_container(
        id=container_name,
        partition_key=PartitionKey(path=partition_key_path),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
    )
    print(f"Container '{container_name}' created successfully.")
except exceptions.CosmosResourceExistsError:
    container = database.get_container_client(container_name)
    print(f"Container '{container_name}' already exists.")

# Room records to insert
rooms = [
    {
        "roomType": "suite",
        "date": "2025-04-12",
        "available": 2,
        "price": "$250",
        "description": "Spacious luxury suite with king-sized bed, ocean view, and elegant decor. Perfect for a romantic getaway."
    },
    {
        "roomType": "double",
        "date": "2025-04-12",
        "available": 4,
        "price": "$150",
        "description": "Comfortable double room with modern design, desk space, and ideal for business travelers or families."
    },
    {
        "roomType": "single",
        "date": "2025-04-12",
        "available": 5,
        "price": "$120",
        "description": "Cozy single room for solo travelers. Includes a reading nook, compact workspace, and courtyard view."
    },
    {
        "roomType": "loft",
        "date": "2025-04-12",
        "available": 3,
        "price": "$300",
        "description": "Stylish open-plan loft with industrial vibes, exposed brick, and a full kitchen. Great for creative retreats."
    },
    {
        "roomType": "penthouse",
        "date": "2025-04-12",
        "available": 1,
        "price": "$500",
        "description": "Premium penthouse suite with skyline views, private balcony, hot tub, and VIP amenities."
    },
    {
        "roomType": "family",
        "date": "2025-04-12",
        "available": 3,
        "price": "$200",
        "description": "Large family suite with two queen beds, kid-friendly decor, and a small play area."
    },
    {
        "roomType": "garden",
        "date": "2025-04-12",
        "available": 2,
        "price": "$180",
        "description": "Peaceful garden-view room with patio access, natural light, and a relaxing atmosphere for reading or yoga."
    },
    {
        "roomType": "executive",
        "date": "2025-04-12",
        "available": 2,
        "price": "$220",
        "description": "Executive suite with private office space, ergonomic chair, espresso machine, and soundproofing for calls."
    },
    {
        "roomType": "accessible",
        "date": "2025-04-12",
        "available": 2,
        "price": "$140",
        "description": "Wheelchair-accessible room with walk-in shower, grab bars, and extra floor space for mobility."
    },
    {
        "roomType": "eco",
        "date": "2025-04-12",
        "available": 2,
        "price": "$160",
        "description": "Eco-friendly room with recycled materials, zero-waste amenities, and views of the green rooftop garden."
    }
]

# Insert room records with vector embeddings
for room in rooms:
    room_id = f"{room['roomType']}_{random.randint(1, 1000)}"
    vector_description = generate_embeddings(room["description"])
    room_record = {
        "id": room_id,
        "roomType": room["roomType"],
        "date": room["date"],
        "available": room["available"],
        "price": room["price"],
        "description": room["description"],
        "vectorDescription": vector_description
    }
    try:
        container.create_item(body=room_record)
        print(f"Inserted room record: {room_id}")
    except exceptions.CosmosHttpResponseError as e:
        print(f"Failed to insert room record {room_id}: {e.message}")

print("All room records have been processed.")
