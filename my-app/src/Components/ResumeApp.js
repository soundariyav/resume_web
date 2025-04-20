import React, { useState } from 'react';

const ResumeApp = () => {
  const [resumeText, setResumeText] = useState('');
  const [jobDesc, setJobDesc] = useState('');
  const [score, setScore] = useState(null);
  const [optimizedResume, setOptimizedResume] = useState('');
  const [category, setCategory] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRelevanceScore = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('resume_text', resumeText);
      formData.append('job_description', jobDesc);

      const response = await fetch('/relevance_score', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setScore(data.relevance_score);
    } catch (err) {
      alert('Error fetching relevance score');
    }
    setLoading(false);
  };

  const handleOptimizeResume = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('existing_resume', resumeText);
      formData.append('job_description', jobDesc);

      const response = await fetch('/optimize_resume', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setOptimizedResume(data.optimized_resume);
    } catch (err) {
      alert('Error optimizing resume');
    }
    setLoading(false);
  };

  const handlePredictCategory = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('resume', resumeText);

      const response = await fetch('/predict-category', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setCategory(data.predicted_category);
    } catch (err) {
      alert('Error predicting category');
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Resume Intelligence Dashboard</h1>

      <div>
        <label>Paste your Resume:</label>
        <textarea
          rows={6}
          value={resumeText}
          onChange={(e) => setResumeText(e.target.value)}
        />

        <label>Paste the Job Description:</label>
        <textarea
          rows={6}
          value={jobDesc}
          onChange={(e) => setJobDesc(e.target.value)}
        />

        <div>
          <button onClick={handleRelevanceScore}>Get Relevance Score</button>
          <button onClick={handleOptimizeResume}>Optimize Resume</button>
          <button onClick={handlePredictCategory}>Predict Job Category</button>
        </div>
      </div>

      {loading && <p>Loading...</p>}

      {score !== null && (
        <div>
          <strong>Relevance Score:</strong> {score}%
        </div>
      )}

      {optimizedResume && (
        <div>
          <strong>Optimized Resume:</strong>
          <pre>{optimizedResume}</pre>
        </div>
      )}

      {category && (
        <div>
          <strong>Predicted Category:</strong> {category}
        </div>
      )}
    </div>
  );
};

export default ResumeApp;
