import os
import shutil
import uuid
import numpy as np
import librosa
import tensorflow.keras as keras
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pymongo import MongoClient
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from fastapi.middleware.cors import CORSMiddleware

# Suppress annoying TensorFlow warnings
app = FastAPI()

# ... (app = FastAPI() should be right above this)

# 1. Explicitly list the exact URLs that are allowed to talk to this server
origins = [
    "http://localhost:5173",  # Your local Vite development server
    "http://localhost:3000",  # Just in case you use standard React
    "https://ai-audio-analyzer-dusky.vercel.app"  # Your live Vercel frontend!
]

# 2. Apply the VIP list to the middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# --- THE DIAGNOSTIC PRINT ---
# Let's ask Python exactly what it sees.
print(f"Looking for .env at: {env_path}")
print(f"API Key loaded: {os.getenv('CLOUDINARY_API_KEY')}")



try:
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client.music_analyzer  # Creates a database called 'music_analyzer'
    feedback_collection = db.feedback_queue  # Creates a collection called 'feedback_queue'
    print("✅ Connected to MongoDB!")
except Exception as e:
    print(f"⚠️ MongoDB Connection Failed: {e}")

    try:
        cloudinary.config( 
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
        api_key = os.getenv("CLOUDINARY_API_KEY"), 
        api_secret = os.getenv("CLOUDINARY_API_SECRET"),
        secure = True
    )
        print("✅ Successfully connected to Cloudinary!")
    except Exception as e:
        print(f"⚠️ Cloudinary Configuration Failed: {e}")

# Initialize the API
app = FastAPI(title="Music Genre Classifier API")

# Allow our future React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # We will lock this down to your Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Brain
MODEL_PATH = "music_genre_cnn_model.keras"
model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("✅ AI Model loaded successfully!")
else:
    print("⚠️ WARNING: Model not found. Make sure your .keras file is in the backend folder.")

# Make sure this matches your dataset (I added 'edm' in here for you!)
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Directories for the Data Flywheel
TEMP_DIR = "temp_uploads"
TRAINING_DATA_DIR = "user_submitted_data"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

def process_audio_full_track(file_path):
    """Chunks the entire song into 3-second segments for the CNN."""
    signal, sr = librosa.load(file_path, sr=22050, duration=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, n_mfcc=13, hop_length=512)
    mfcc = mfcc.T
    
    expected_length = 130
    num_chunks = len(mfcc) // expected_length
    if num_chunks == 0:
        return None
        
    chunks = []
    for i in range(num_chunks):
        start = i * expected_length
        end = start + expected_length
        chunks.append(mfcc[start:end])
        
    batch_features = np.array(chunks)
    batch_features = batch_features[..., np.newaxis]
    return batch_features
@app.get("/ping")
def ping_server():
    return {"status": "The Render server is wide awake!"}
@app.post("/predict/")
def predict_genre(file: UploadFile = File(...)):
    print(f"🚨 FRONT DOOR BREACHED: Received file {file.filename}!")
    if model is None:
        return {"error": "Model not loaded on the server."}

    # Generate a unique ID for this specific upload
    file_id = f"{uuid.uuid4()}_{file.filename}"
    temp_file_path = os.path.join(TEMP_DIR, file_id)

    # Save the file temporarily
    with open(temp_file_path, "wb") as buffer:
        buffer.write(file.file.read())

    try:
        features = process_audio_full_track(temp_file_path)
        if features is None:
            return {"error": "Audio clip too short! Needs to be at least 3 seconds."}
            
        # Get predictions for all chunks and average them
        predictions = model.predict(features, verbose=0)
        average_prediction = np.mean(predictions, axis=0)
        
        # Grab the top 3 predictions
        top_3_indices = np.argsort(average_prediction)[::-1][:3]
        top_3_results = []
        for i in top_3_indices:
            top_3_results.append({
                "genre": GENRES[i],
                "confidence": round(float(average_prediction[i]) * 100, 2)
            })

        return {
            "file_id": file_id,
            "predictions": top_3_results
        }
    except Exception as e:
        return {"error": str(e)}

# --- THE HEAVY LIFTING FUNCTION (Runs invisibly in the background) ---
def process_upload_and_log(file_id: str, clean_genre: str, temp_file_path: str):
    try:
        print(f"☁️ Background Task: Uploading {file_id} to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            temp_file_path, 
            resource_type="video",
            folder=f"music_analyzer/{clean_genre}",
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET")
        )
        
        cloud_url = upload_result.get("secure_url")
        print(f"✅ Background Task: Upload complete! URL: {cloud_url}")

        feedback_collection.insert_one({
            "file_id": file_id,
            "genre": clean_genre,
            "cloud_url": cloud_url,
            "public_id": upload_result.get("public_id"),
            "status": "pending_training"
        })
    except Exception as e:
        print(f"❌ Background Task Error: {e}")
    finally:
        # Always delete the temp file when finished, even if upload fails
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- THE USER-FACING ENDPOINT (Lightning fast) ---
@app.post("/submit-feedback/")
def submit_feedback(
    background_tasks: BackgroundTasks, # <-- Inject the background task manager
    file_id: str = Form(...), 
    true_genre: str = Form(...)
):
    clean_genre = true_genre.lower().strip().replace(" ", "_")
    temp_file_path = os.path.join(TEMP_DIR, file_id)
    
    if not os.path.exists(temp_file_path):
        return {"error": "Original file lost or session expired."}
        
    # Hand the heavy lifting off to the background task!
    background_tasks.add_task(process_upload_and_log, file_id, clean_genre, temp_file_path)
    
    # Return success to React INSTANTLY so the UI doesn't hang
    return {"message": f"Awesome! '{clean_genre}' registered. Thanks for your help teaching the AI!"}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)