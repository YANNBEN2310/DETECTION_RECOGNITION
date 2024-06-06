function startScanning() {
  const videoPlayer = document.getElementById('videoPlayer');
  const imageDisplay = document.getElementById('imageDisplay');

  if (!videoPlayer.classList.contains('hidden')) {
    videoPlayer.play();
    // Add your deepfake detection logic here
  } else if (!imageDisplay.classList.contains('hidden')) {
    // Add your deepfake detection logic here
  } else {
    alert('Please load a video or image first.');
  }
}
