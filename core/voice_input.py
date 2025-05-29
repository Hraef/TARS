## handles voice input and processing
## loads whisper
## transcribes audio to text

import whisper
import os
import wave
import tempfile
import threading
from typing import Optional
import contextlib
import io

# Simple error suppression for PyAudio import
@contextlib.contextmanager
def suppress_stderr():
    """Simple context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

# Import PyAudio with error suppression
try:
    with suppress_stderr():
        import pyaudio
except ImportError as e:
    print(f"❌ PyAudio not installed: {e}")
    print("💡 Install with: pip install pyaudio")
    raise
except Exception as e:
    print(f"⚠️  PyAudio import warning (likely harmless): {e}")
    import pyaudio  # Try importing without suppression


class VoiceInput:
    """Handles voice input and processing using Whisper for transcription."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the voice input system.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.model = None
        self.is_recording = False
        self.audio_data = []
        
        # Audio configuration
        self.sample_rate = 16000  # Whisper's native sample rate
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # PyAudio instance with error suppression
        with suppress_stderr():
            self.audio = pyaudio.PyAudio()
        self.stream = None
        
        print(f"VoiceInput initialized with {model_size} model")
    
    def load_model(self):
        """Load the Whisper model. Called lazily to avoid startup delays."""
        if self.model is None:
            print(f"Loading Whisper {self.model_size} model...")
            try:
                import torch
                
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                    print("⚠️  No CUDA GPU detected, using CPU mode.")
                
                print(f"🔄 Loading model on device: {device}")
                self.model = whisper.load_model(self.model_size, device=device)
                print("✅ Whisper model loaded successfully!")
                
                # Display memory info if using GPU
                if device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {total_memory:.2f}GB")
                
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                print("🔄 Falling back to CPU mode...")
                try:
                    self.model = whisper.load_model(self.model_size, device="cpu")
                    print("✅ Whisper model loaded successfully on CPU!")
                except Exception as cpu_error:
                    print(f"❌ Failed to load model on CPU: {cpu_error}")
                    raise
    
    def start_recording(self):
        """Start recording audio from the microphone."""
        if self.is_recording:
            print("Already recording!")
            return
        
        try:
            with suppress_stderr():
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
            
            self.is_recording = True
            self.audio_data = []
            print("🎤 Recording started... Press Enter to stop.")
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
    
    def _record_audio(self):
        """Internal method to record audio data."""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_data.append(data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
    
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            print("Not currently recording!")
            return
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        
        # Close the audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        print("🛑 Recording stopped.")
    
    def save_audio_to_file(self, filename: Optional[str] = None) -> str:
        """
        Save recorded audio data to a WAV file.
        
        Args:
            filename: Optional filename. If None, creates a temporary file.
            
        Returns:
            Path to the saved audio file.
        """
        if not self.audio_data:
            raise ValueError("No audio data to save!")
        
        if filename is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            filename = temp_file.name
            temp_file.close()
        
        # Save audio data as WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.audio_data))
        
        return filename
    
    def transcribe_audio(self, audio_file: Optional[str] = None) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_file: Path to audio file. If None, uses recorded audio data.
            
        Returns:
            Transcribed text.
        """
        # Load model if not already loaded
        self.load_model()
        
        # If no audio file provided, save current recording
        if audio_file is None:
            if not self.audio_data:
                raise ValueError("No audio data available for transcription!")
            audio_file = self.save_audio_to_file()
            cleanup_file = True
        else:
            cleanup_file = False
        
        try:
            print("🔄 Transcribing audio...")
            result = self.model.transcribe(audio_file)
            transcribed_text = result["text"].strip()
            
            print(f"📝 Transcription: '{transcribed_text}'")
            return transcribed_text
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""
        
        finally:
            # Clean up temporary file if we created it
            if cleanup_file and os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def record_and_transcribe(self) -> str:
        """
        Convenience method to record audio and transcribe it in one call.
        This is a blocking method that waits for user input to stop recording.
        
        Returns:
            Transcribed text.
        """
        self.start_recording()
        
        # Wait for user to press Enter
        input()  # This will block until user presses Enter
        
        self.stop_recording()
        return self.transcribe_audio()
    
    def cleanup(self):
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
        
        print("VoiceInput cleanup completed.")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass

