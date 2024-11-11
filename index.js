function runOpenCV() {
    // Capture the image from the user's camera
    fetch('/capture_image', {
      method: 'POST',
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        updatePredictionResult(data.error);
      } else {
        // Update the prediction result with the emotion and emotion code
        updatePredictionResult(`Emotion: ${data.emotion}, Emotion Code: ${data.emotion_code}, Probability: ${data.max_prob.toFixed(2)}`);
      }
    })
    .catch(error => {
      console.error('Error in runOpenCV:', error);
      updatePredictionResult('Error capturing image. Please try again.');
    });
  }