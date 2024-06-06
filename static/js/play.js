function loadMedia(input) {
  const file = input.files[0];
  if (!file) return;

  const videoPlayer = document.getElementById('videoPlayer');
  const videoSource = document.getElementById('videoSource');
  const imageDisplay = document.getElementById('imageDisplay');

  const fileType = file.type.split('/')[0];
  const fileExtension = file.type.split('/')[1];

  const videoExtensions = ['mp4', 'webm', 'ogg'];
  const imageExtensions = ['jpeg', 'png', 'gif', 'bmp', 'webp'];

  const reader = new FileReader();
  reader.onload = function(e) {
    if (fileType === 'video' && videoExtensions.includes(fileExtension)) {
      videoSource.src = e.target.result;
      videoSource.type = file.type;
      videoPlayer.load();
      videoPlayer.classList.remove('hidden');
      imageDisplay.classList.add('hidden');
    } else if (fileType === 'image' && imageExtensions.includes(fileExtension)) {
      imageDisplay.src = e.target.result;
      imageDisplay.classList.remove('hidden');
      videoPlayer.classList.add('hidden');
    } else {
      alert('Unsupported file type. Please upload a valid video or image file.');
    }
  };
  reader.readAsDataURL(file);
}
