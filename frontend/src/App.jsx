import { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [customGenre, setCustomGenre] = useState('');
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null);
    setFeedbackStatus('');
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an audio file first!");

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://aarlo28-ai-sound-analyzer.hf.space/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      
      if (data.error) alert(data.error);
      else {
        setResults(data.predictions);
        setFileId(data.file_id);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to connect to the server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedback = async (trueGenre) => {
    if (!trueGenre) return;
    
    setIsSubmittingFeedback(true);

    const formData = new FormData();
    formData.append("file_id", fileId);
    formData.append("true_genre", trueGenre);

    try {
      const response = await fetch("https://aarlo28-ai-sound-analyzer.hf.space/submit-feedback/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setFeedbackStatus(data.message || data.error);
    } catch (error) {
      console.error("Error:", error);
      setFeedbackStatus("Failed to submit feedback.");
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  return (
    <div className="app-container">
      
      <header className="header">
        <h1>🎵 AI Music Genre Analyzer</h1>
        <p>Upload a track and let the neural network guess the genre.</p>
      </header>

      <div className="upload-section">
        <input type="file" accept="audio/*" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!file || isLoading}>
          {isLoading ? "Analyzing Audio..." : "Predict Genre"}
        </button>
      </div>

      {results && (
        <div className="results-card">
          <h2>🎧 AI Predictions</h2>
          <ul className="predictions-list">
            {results.map((item, index) => (
              <li key={item.genre} className={index === 0 ? "prediction-item winner" : "prediction-item runner-up"}>
                {index === 0 ? `🏆 ${item.genre.toUpperCase()}` : `#${index + 1}: ${item.genre.toUpperCase()}`} ({item.confidence}%)
              </li>
            ))}
          </ul>

          {!feedbackStatus ? (
            <div className="feedback-section">
              <h3>Was it wrong? Help the AI Learn!</h3>
              <p>Which genre is this actually?</p>
              
              <div className="feedback-buttons">
                {results.map((item) => (
                  <button 
                    key={item.genre} 
                    onClick={() => handleFeedback(item.genre)}
                    disabled={isSubmittingFeedback}
                  >
                    {isSubmittingFeedback ? "Saving..." : `It's ${item.genre}`}
                  </button>
                ))}
              </div>
              
              <div className="custom-genre-form">
                <input 
                  type="text" 
                  placeholder="Type a custom genre..." 
                  value={customGenre}
                  onChange={(e) => setCustomGenre(e.target.value)}
                  disabled={isSubmittingFeedback}
                />
                <button 
                  onClick={() => handleFeedback(customGenre)}
                  disabled={isSubmittingFeedback || !customGenre}
                >
                  Submit
                </button>
              </div>
            </div>
          ) : (
            <div className="success-message">
              ✅ {feedbackStatus}
            </div>
          )}
        </div>
      )}

      {/* Footer Signature */}
      <footer className="footer-signature">
        <p>
          Built and Maintained by{" "}
          <a 
            href="https://in.linkedin.com/in/aarush-lohani-691305261" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Aarush Lohani
          </a>
        </p>
      </footer>

    </div>
  );
}

export default App;