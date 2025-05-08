document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const imageFolderInput = document.getElementById('imageFolder');
    const yamlPathInput = document.getElementById('yamlPath');
    const videoFeed = document.getElementById('videoFeed');
    const processingStatusSpan = document.getElementById('processingStatus');
    const fpsDisplaySpan = document.getElementById('fpsDisplay');

    let statusInterval = null; // To store the interval for polling status

    // Function to update the status and FPS display
    function updateStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                processingStatusSpan.textContent = data.is_processing ? 'Processing...' : 'Idle';
                fpsDisplaySpan.textContent = data.current_fps.toFixed(2); // Display FPS with 2 decimal places

                // If processing stops, clear the interval
                if (!data.is_processing && statusInterval) {
                     clearInterval(statusInterval);
                     statusInterval = null;
                     stopButton.disabled = true;
                     startButton.disabled = false;
                     videoFeed.src = ""; // Stop the video feed
                     processingStatusSpan.textContent = "Finished";
                     fpsDisplaySpan.textContent = "N/A";
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
                // If there's an error fetching status, assume processing stopped or failed
                if (statusInterval) {
                    clearInterval(statusInterval);
                    statusInterval = null;
                }
                stopButton.disabled = true;
                startButton.disabled = false;
                videoFeed.src = ""; // Stop the video feed
                processingStatusSpan.textContent = "Error/Stopped";
                fpsDisplaySpan.textContent = "N/A";
            });
    }

    startButton.addEventListener('click', function() {
        const imageFolder = imageFolderInput.value;
        const yamlPath = yamlPathInput.value;

        if (!imageFolder || !yamlPath) {
            alert("Please enter both the image folder path and the YAML path.");
            return;
        }

        // Use FormData to send the paths
        const formData = new FormData();
        formData.append('image_folder', imageFolder);
        formData.append('yaml_path', yamlPath);

        // Disable start button, enable stop button
        startButton.disabled = true;
        stopButton.disabled = false;
        processingStatusSpan.textContent = "Starting...";
        fpsDisplaySpan.textContent = "N/A";
        videoFeed.src = ""; // Clear previous feed

        fetch('/start_processing', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log(data.message);
                // Set the video feed source to the MJPEG stream URL
                videoFeed.src = "{{ url_for('video_feed') }}?" + new Date().getTime(); // Add timestamp to prevent caching
                // Start polling for status updates
                if (!statusInterval) {
                    statusInterval = setInterval(updateStatus, 500); // Poll every 500ms
                }
                updateStatus(); // Initial status update
            } else {
                console.error('Error starting processing:', data.message);
                alert('Error starting processing: ' + data.message);
                startButton.disabled = false; // Re-enable start button on error
                stopButton.disabled = true;
                processingStatusSpan.textContent = "Idle";
                fpsDisplaySpan.textContent = "N/A";
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            alert('An error occurred while trying to start processing.');
            startButton.disabled = false; // Re-enable start button on error
            stopButton.disabled = true;
            processingStatusSpan.textContent = "Idle";
            fpsDisplaySpan.textContent = "N/A";
        });
    });

    stopButton.addEventListener('click', function() {
        // Disable stop button
        stopButton.disabled = true;
        processingStatusSpan.textContent = "Stopping...";

        fetch('/stop_processing', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
             // Status update interval will catch the 'Idle' state and re-enable start
        })
        .catch(error => {
            console.error('Fetch error:', error);
            alert('An error occurred while trying to stop processing.');
             // Re-enable stop button if the stop request itself failed
            stopButton.disabled = false;
        });
    });

     // Initial status check when the page loads
    updateStatus();
});
