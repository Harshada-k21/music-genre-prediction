import os
import subprocess
import requests
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Load Configurations
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client.music_analyzer
queue = db.feedback_queue

# Setup Cloudinary
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

MASTER_DATA_DIR = os.path.join(BASE_DIR, "genres_original")

def run_purge():
    print("🚀 Initiating Cloud Retraining & Purge Sequence...")
    
    # 1. Check the queue
    pending_files = list(queue.find({"status": "pending_training"}))
    if not pending_files:
        print("🤷‍♂️ No new user submissions found in the database. Exiting.")
        return

    print(f"📥 Found {len(pending_files)} new tracks in the cloud queue.")

    # 2. Download from Cloudinary & Delete to save space
    for record in pending_files:
        genre = record['genre']
        file_id = record['file_id']
        cloud_url = record.get('cloud_url')
        public_id = record.get('public_id')
        
        dst_folder = os.path.join(MASTER_DATA_DIR, genre)
        dst_path = os.path.join(dst_folder, file_id)
        os.makedirs(dst_folder, exist_ok=True)
        
        if cloud_url and public_id:
            print(f"⬇️ Downloading {file_id} from cloud...")
            try:
                # Download the audio file
                response = requests.get(cloud_url)
                response.raise_for_status() # Check for download errors
                
                with open(dst_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Saved to master dataset: {genre}/{file_id}")
                
                # Delete from Cloudinary to free up free-tier storage!
                print(f"🗑️ Deleting {file_id} from Cloudinary to save space...")
                cloudinary.uploader.destroy(public_id, resource_type="video")
                
            except Exception as e:
                print(f"⚠️ Failed to process {file_id}: {e}")
                continue

    # 3. Trigger Feature Extraction
    print("\n⚙️ Running feature extraction (preprocess.py)... This may take a while.")
    try:
        subprocess.run(["python", "preprocess.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Feature extraction failed! Aborting purge.")
        return

    # 4. Trigger Model Training
    print("\n🧠 Training the new Neural Network (train.py)...")
    try:
        subprocess.run(["python", "train.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Model training failed! Aborting purge.")
        return

    # 5. Empty the Database
    print("\n🧹 Emptying the MongoDB Queue...")
    queue.delete_many({})
    
    print("\n✨ Purge Complete! The AI is smarter, and your cloud storage is empty!")

if __name__ == "__main__":
    run_purge()