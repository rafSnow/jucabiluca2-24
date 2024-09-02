document.getElementById('uploadForm').addEventListener('submit', function(event) {
  event.preventDefault();
  
  const formData = new FormData(this);
  
  fetch('/upload', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      document.getElementById('result').textContent = JSON.stringify(data, null, 2);
  })
  .catch(error => {
      console.error('Error:', error);
  });
});
