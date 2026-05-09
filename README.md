# 🎵 Music Genre Prediction using CNN

A full-stack Machine learning web application that classifies music genres from audio files using a Convolutional Neural Network (CNN) and spectrogram-based feature extraction.

---

## 🚀 Features

- Upload audio files for genre prediction  
- CNN-based deep learning model for classification  
- Audio preprocessing using Librosa  
- Real-time prediction via backend API  
- Modern React + Vite frontend  
- Docker support for deployment  

---

## 🛠️ Tech Stack

### Frontend
- React
- Vite
- JavaScript
- CSS

### Backend
- Python
- Flask / FastAPI
- TensorFlow / Keras
- Librosa

### Others
- Docker
- REST API

---

## 📁 Project Structure

```
backend/
  main.py
  train.py
  test.py
  preprocess.py
  purge.py
  requirements.txt
  Dockerfile
  music_genre_cnn_model.keras
  user_submitted_data/

frontend/
  src/
  public/
  package.json
  vite.config.js
  index.html

.gitignore
README.md
```

---

## ⚙️ How to Run

### 🔹 Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

---

### 🔹 Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🧠 Model Training

```bash
python train.py
```

---

## 📊 Dataset

- GTZAN Music Genre Dataset (or your dataset)
- Audio converted into spectrograms for CNN input

## 🚀 Future Improvements

- Real-time audio classification  
- Improve model accuracy  
- Cloud deployment (AWS / Render / Vercel)  
- Mobile application version  

---

## 👨‍💻 Author

- Harshada K  

---

## ⭐ Note

If you like this project, consider giving it a star ⭐ on GitHub!
