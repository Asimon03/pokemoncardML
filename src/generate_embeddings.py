from pymongo import MongoClient
from embed import embed_images

from PIL import Image
from io import BytesIO
import requests

# MongoDB Atlas connection string
uri = "CONNECTION STRING"

client = MongoClient(uri)

db = client["pokemon_db"]
collection = db["cards"]

processed = 0
skipped = 0

# Process cards that don't already have embeddings
cards = collection.find({"embedding": {"$exists": False}})

for card in cards:
    try:
        print(f"Processing: {card['name']} ({card['id']})")

        # Download the Pokemon Card image
        response = requests.get(card["image_url"], timeout=30)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Generate embedding
        embedding = embed_images([image])[0]

        # Store embedding
        collection.update_one(
            {"_id": card["_id"]},
            {
                "$set": {
                    "embedding": embedding.tolist()
                }
            }
        )

        processed += 1

        # Progress message every 100 cards
        if processed % 100 == 0:
            print(f"Processed {processed} cards...")

    except Exception as e:
        print(f"Error processing {card['name']}: {e}")
        skipped += 1

print("\nFinished!")
print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
