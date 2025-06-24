// AutoMix Web Application JavaScript

class AutoMixApp {
    constructor() {
        this.vocalFile = null;
        this.bgmFile = null;
        this.vocalPath = null;
        this.bgmPath = null;
        this.outputPath = null;
        this.outputFilename = null;
        this.statusInterval = null;
        
        this.init();
    }
    
    init() {
        this.setupFileInputs();
        this.setupSliders();
        this.setupProcessButton();
        this.loadPlugins();
    }
    
    setupFileInputs() {
        // Vocal file input
        const vocalInput = document.getElementById('vocal-input');
        const vocalDrop = document.getElementById('vocal-drop');
        const vocalFilename = document.getElementById('vocal-filename');
        
        vocalInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], 'vocal', vocalFilename);
            }
        });
        
        this.setupDropZone(vocalDrop, (file) => {
            this.handleFileSelect(file, 'vocal', vocalFilename);
            vocalInput.files = this.createFileList([file]);
        });
        
        // BGM file input
        const bgmInput = document.getElementById('bgm-input');
        const bgmDrop = document.getElementById('bgm-drop');
        const bgmFilename = document.getElementById('bgm-filename');
        
        bgmInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0], 'bgm', bgmFilename);
            }
        });
        
        this.setupDropZone(bgmDrop, (file) => {
            this.handleFileSelect(file, 'bgm', bgmFilename);
            bgmInput.files = this.createFileList([file]);
        });
    }
    
    setupDropZone(dropZone, onDrop) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            });
        });
        
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                onDrop(files[0]);
            }
        });
    }
    
    createFileList(files) {
        const dataTransfer = new DataTransfer();
        files.forEach(file => dataTransfer.items.add(file));
        return dataTransfer.files;
    }
    
    async handleFileSelect(file, type, filenameElement) {
        if (!this.isValidAudioFile(file)) {
            this.showError('Invalid file type. Please select an audio file.');
            return;
        }
        
        // Update UI
        filenameElement.textContent = file.name;
        filenameElement.classList.add('show');
        
        // Upload file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                if (type === 'vocal') {
                    this.vocalFile = file;
                    this.vocalPath = data.path;
                } else {
                    this.bgmFile = file;
                    this.bgmPath = data.path;
                }
                
                this.checkReadyToProcess();
            } else {
                this.showError(data.error || 'Failed to upload file');
            }
        } catch (error) {
            this.showError('Failed to upload file: ' + error.message);
        }
    }
    
    isValidAudioFile(file) {
        const validExtensions = ['wav', 'mp3', 'm4a', 'flac', 'ogg'];
        const extension = file.name.split('.').pop().toLowerCase();
        return validExtensions.includes(extension);
    }
    
    setupSliders() {
        // Vocal volume slider
        const vocalSlider = document.getElementById('vocal-volume');
        const vocalValue = document.getElementById('vocal-volume-value');
        
        vocalSlider.addEventListener('input', (e) => {
            vocalValue.textContent = e.target.value;
        });
        
        // BGM volume slider
        const bgmSlider = document.getElementById('bgm-volume');
        const bgmValue = document.getElementById('bgm-volume-value');
        
        bgmSlider.addEventListener('input', (e) => {
            bgmValue.textContent = e.target.value;
        });
    }
    
    setupProcessButton() {
        const processBtn = document.getElementById('process-btn');
        processBtn.addEventListener('click', () => this.process());
    }
    
    checkReadyToProcess() {
        const processBtn = document.getElementById('process-btn');
        processBtn.disabled = !(this.vocalPath && this.bgmPath);
    }
    
    async loadPlugins() {
        try {
            const response = await fetch('/api/plugins');
            const plugins = await response.json();
            
            const container = document.getElementById('plugins-container');
            container.innerHTML = '';
            
            plugins.forEach(plugin => {
                const pluginEl = document.createElement('div');
                pluginEl.className = 'plugin-item';
                pluginEl.innerHTML = `
                    <div class="plugin-info">
                        <h4>${plugin.name}</h4>
                        <p>${plugin.description}</p>
                    </div>
                    <label class="checkbox-group">
                        <input type="checkbox" data-plugin="${plugin.name}" ${plugin.enabled ? 'checked' : ''}>
                        <span>Enable</span>
                    </label>
                `;
                container.appendChild(pluginEl);
            });
        } catch (error) {
            console.error('Failed to load plugins:', error);
        }
    }
    
    async process() {
        if (!this.vocalPath || !this.bgmPath) {
            this.showError('Please select both vocal and BGM files');
            return;
        }
        
        // Get settings
        const settings = {
            vocal_path: this.vocalPath,
            bgm_path: this.bgmPath,
            preset: document.getElementById('preset').value,
            vocal_volume: parseFloat(document.getElementById('vocal-volume').value),
            bgm_volume: parseFloat(document.getElementById('bgm-volume').value),
            reverb: document.getElementById('reverb').value || null,
            video_template: document.getElementById('video-template').value,
            audio_only: document.getElementById('audio-only').checked,
            chunk_processing: document.getElementById('chunk-processing').checked,
            streaming: document.getElementById('streaming').checked,
            preview_mode: document.getElementById('preview-mode').checked
        };
        
        // Disable process button
        const processBtn = document.getElementById('process-btn');
        processBtn.disabled = true;
        
        // Show progress
        this.showProgress();
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Start monitoring progress
                this.startStatusMonitoring();
            } else {
                this.showError(data.error || 'Failed to start processing');
                processBtn.disabled = false;
            }
        } catch (error) {
            this.showError('Failed to start processing: ' + error.message);
            processBtn.disabled = false;
        }
    }
    
    startStatusMonitoring() {
        this.statusInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                if (status.is_processing) {
                    this.updateProgress(status.progress, status.message);
                } else {
                    clearInterval(this.statusInterval);
                    
                    if (status.error) {
                        this.showError(status.error);
                    } else if (status.output_path) {
                        this.outputPath = status.output_path;
                        this.outputFilename = status.output_filename;
                        this.showSuccess();
                    }
                    
                    // Re-enable process button
                    document.getElementById('process-btn').disabled = false;
                }
            } catch (error) {
                console.error('Failed to get status:', error);
            }
        }, 500);
    }
    
    showProgress() {
        document.getElementById('progress-container').style.display = 'block';
        document.getElementById('result-container').style.display = 'none';
        document.getElementById('error-container').style.display = 'none';
    }
    
    updateProgress(percent, message) {
        document.getElementById('progress-fill').style.width = percent + '%';
        document.getElementById('progress-text').textContent = message;
    }
    
    showSuccess() {
        document.getElementById('progress-container').style.display = 'none';
        document.getElementById('result-container').style.display = 'block';
        document.getElementById('error-container').style.display = 'none';
        
        // Setup download button
        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.onclick = () => {
            window.location.href = `/api/download/${this.outputFilename}`;
        };
    }
    
    showError(message) {
        document.getElementById('progress-container').style.display = 'none';
        document.getElementById('result-container').style.display = 'none';
        document.getElementById('error-container').style.display = 'block';
        document.getElementById('error-message').textContent = message;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AutoMixApp();
});