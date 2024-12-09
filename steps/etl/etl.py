import json
import os

import feedparser
import pymongo
import requests
import youtube_dl
from clearml import PipelineController, Task

# MongoDB connection
client = pymongo.MongoClient("mongodb://mongodb:27017/")
db = client["rag_data"]
collection = db["raw_data"]

# Task and Pipeline Setup
task = Task.init(project_name="RAG System", task_name="ETL Pipeline")
pipeline = PipelineController(name="ETL_Pipeline", project="RAG System")

# Extract Step: Fetch GitHub ROS2 Documentation URLs
def extract_github_data():
    github_api_url = "https://api.github.com/repos/ros2/ros2/releases"
    response = requests.get(github_api_url)
    response.raise_for_status()
    releases = response.json()
    data = [{"url": release["html_url"], "source": "github"} for release in releases]
    collection.insert_many(data)
    print(f"Fetched {len(data)} GitHub release URLs.")

pipeline.add_function_step(
    name="Extract_GitHub_Data",
    function=extract_github_data,
    execution_queue="default",
)

# Extract Step: Fetch YouTube Videos
def extract_youtube_videos():
    youtube_query = "ROS2 tutorials"
    ydl_opts = {"quiet": True, "simulate": True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch10:{youtube_query}", download=False)
    videos = [{"url": entry["webpage_url"], "source": "youtube"} for entry in result["entries"]]
    collection.insert_many(videos)
    print(f"Fetched {len(videos)} YouTube video URLs.")

pipeline.add_function_step(
    name="Extract_YouTube_Data",
    function=extract_youtube_videos,
    execution_queue="default",
)

# Extract Step: Fetch LinkedIn Posts
def extract_linkedin_data():
    linkedin_api_url = "https://api.linkedin.com/v2/articles"
    headers = {
        "Authorization": f"Bearer {os.getenv('LINKEDIN_ACCESS_TOKEN')}"
    }
    response = requests.get(linkedin_api_url, headers=headers)
    response.raise_for_status()
    articles = response.json().get("elements", [])
    data = [{"url": article["url"], "source": "linkedin"} for article in articles]
    collection.insert_many(data)
    print(f"Fetched {len(data)} LinkedIn articles.")

pipeline.add_function_step(
    name="Extract_LinkedIn_Data",
    function=extract_linkedin_data,
    execution_queue="default",
)

# Extract Step: Fetch Medium Articles
def extract_medium_data():
    medium_rss_feed = "https://medium.com/feed/topic/machine-learning"
    feed = feedparser.parse(medium_rss_feed)
    articles = [
        {"url": entry.link, "title": entry.title, "source": "medium"}
        for entry in feed.entries
    ]
    collection.insert_many(articles)
    print(f"Fetched {len(articles)} Medium articles.")

pipeline.add_function_step(
    name="Extract_Medium_Data",
    function=extract_medium_data,
    execution_queue="default",
)

# Transform Step: Process Raw Data
def transform_data():
    raw_data = list(collection.find())
    processed_data = []
    for item in raw_data:
        # Add transformation logic if required
        item["processed"] = True
        processed_data.append(item)
    collection.delete_many({})
    collection.insert_many(processed_data)
    print(f"Processed {len(processed_data)} items.")

pipeline.add_function_step(
    name="Transform_Data",
    function=transform_data,
    execution_queue="default",
)

# Load Step: Store Data in MongoDB
def load_data():
    processed_data = list(collection.find({"processed": True}))
    print(f"Loaded {len(processed_data)} items into MongoDB.")
    for item in processed_data:
        print(f"Stored URL: {item['url']}")

pipeline.add_function_step(
    name="Load_Data",
    function=load_data,
    execution_queue="default",
)

# Run Pipeline
if __name__ == "__main__":
    pipeline.set_default_execution_queue("default")
    pipeline.start()
    pipeline.wait()
