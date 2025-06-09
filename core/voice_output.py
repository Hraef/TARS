## speaks back the text using pyttsx3

import pyttsx3
import subprocess
import os
import threading
import queue
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class VoiceOutput:
    """Handles LLM processing and text-to-speech output for TARS."""
    
    def __init__(self, model: str = "mistral", voice_rate: int = 180, voice_volume: float = 0.9, tars_voice: bool = True):
        """
        Initialize the voice output system with LLM integration.
        
        Args:
            model: Ollama model to use for LLM processing
            voice_rate: Speech rate (words per minute) - Default 180 for TARS-like speech
            voice_volume: Voice volume (0.0 to 1.0)
            tars_voice: Enable TARS-specific voice optimizations
        """
        print("🔊 Initializing TARS Voice Output System...")
        
        # LLM Configuration
        self.model = model
        self.ollama_path = os.getenv("OLLAMA_PATH", "ollama")
        
        # Text-to-Speech Configuration
        self.tts_engine = None
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.tars_voice = tars_voice
        
        # Threading for non-blocking operations
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.is_speaking = False
        self.shutdown_flag = False
        
        # Initialize TTS engine
        self._init_tts_engine()
        
        # Start TTS worker thread
        self._start_tts_worker()
        
        print("✅ Voice Output System initialized!")
    
    def _configure_tars_voice(self):
        """Configure TARS-specific voice settings for a more robotic sound."""
        try:
            # TARS voice characteristics
            tars_rate = max(150, min(self.voice_rate, 190))  # Slower, more deliberate
            tars_volume = self.voice_volume
            
            self.tts_engine.setProperty('rate', tars_rate)
            self.tts_engine.setProperty('volume', tars_volume)
            
            print(f"🤖 TARS voice configured: Rate={tars_rate}, Volume={tars_volume}")
            
        except Exception as e:
            print(f"⚠️  TARS voice configuration warning: {e}")
    
    def _init_tts_engine(self):
        """Initialize the text-to-speech engine with WSL2 optimizations."""
        try:
            # Try different TTS drivers for better WSL2 compatibility
            drivers_to_try = ['espeak', 'sapi5', 'nsss', 'dummy']
            
            for driver in drivers_to_try:
                try:
                    print(f"🔄 Trying TTS driver: {driver}")
                    if driver == 'dummy':
                        # Last resort - dummy driver for testing
                        self.tts_engine = pyttsx3.init(driverName='dummy')
                    else:
                        self.tts_engine = pyttsx3.init(driverName=driver)
                    
                    print(f"✅ Successfully initialized with {driver} driver")
                    break
                    
                except Exception as driver_error:
                    print(f"⚠️  {driver} driver failed: {driver_error}")
                    continue
            
            if not self.tts_engine:
                # Final fallback - try default initialization
                print("🔄 Trying default TTS initialization...")
                self.tts_engine = pyttsx3.init()
            
            # Configure voice properties if engine is initialized
            if self.tts_engine:
                try:
                    # Apply TARS-specific voice settings
                    if self.tars_voice:
                        self._configure_tars_voice()
                    else:
                        self.tts_engine.setProperty('rate', self.voice_rate)
                        self.tts_engine.setProperty('volume', self.voice_volume)
                    
                    # Try to set a more robotic/AI-like voice if available
                    voices = self.tts_engine.getProperty('voices')
                    if voices and len(voices) > 0:
                        # Prefer deep male voices for TARS character
                        tars_voice = None
                        for voice in voices:
                            voice_name = getattr(voice, 'name', '').lower()
                            # Look for deeper, more robotic sounding voices
                            if any(keyword in voice_name for keyword in ['male', 'david', 'mark', 'deep', 'low', 'bass']):
                                tars_voice = voice.id
                                break
                        
                        if tars_voice:
                            self.tts_engine.setProperty('voice', tars_voice)
                            print(f"🎭 Using TARS voice: {tars_voice}")
                        else:
                            print("🎭 Using default voice (TARS optimized)")
                    
                    print("🔊 TARS voice system configured successfully")
                    
                except Exception as config_error:
                    print(f"⚠️  Voice configuration warning: {config_error}")
                    # Engine is still usable even if configuration fails
            
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {e}")
            print("💡 Falling back to silent mode - LLM responses will be text-only")
            self.tts_engine = None
    
    def _start_tts_worker(self):
        """Start the TTS worker thread for non-blocking speech."""
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
    
    def _tts_worker(self):
        """Worker thread for handling TTS queue."""
        while not self.shutdown_flag:
            try:
                # Get text from queue with timeout
                text = self.tts_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                
                # Speak the text
                self._speak_text(text)
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in TTS worker: {e}")
    
    def _speak_text(self, text: str):
        """Internal method to speak text using TTS engine."""
        if not text.strip():
            return
        
        if not self.tts_engine:
            print(f"💬 TARS says (text-only): {text}")
            return
        
        try:
            self.is_speaking = True
            print(f"🗣️  TARS speaking: {text}")
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            print(f"💬 TARS says (fallback): {text}")
            # Try alternative speaking method with TARS-optimized eSpeak
            try:
                import subprocess
                if self.tars_voice:
                    # TARS-optimized eSpeak parameters
                    subprocess.run([
                        'espeak', 
                        '-s', '170',    # Speed: 170 words per minute (slower)
                        '-p', '20',     # Pitch: Lower pitch (0-99, default ~50)
                        '-a', '100',    # Amplitude: Volume
                        '-g', '5',      # Gap between words (slight pause)
                        '-v', 'en+m3',  # Voice variant (male voice 3)
                        text
                    ], capture_output=True, timeout=10)
                else:
                    subprocess.run(['espeak', text], capture_output=True, timeout=10)
            except:
                pass  # Silent fallback to text-only
        finally:
            self.is_speaking = False
    
    def query_llm(self, prompt: str, timeout: int = 60) -> str:
        """
        Query the local LLM with the given prompt.
        
        Args:
            prompt: The input prompt for the LLM
            timeout: Timeout in seconds for LLM response
            
        Returns:
            LLM response text
        """
        try:
            print(f"🧠 Querying LLM with: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Enhance prompt with TARS personality
            enhanced_prompt = self._enhance_prompt_with_personality(prompt)
            
            result = subprocess.run(
                [self.ollama_path, "run", self.model],
                input=enhanced_prompt,
                text=True,
                encoding='utf-8',
                errors="replace",
                capture_output=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"✅ LLM Response received: '{response[:100]}{'...' if len(response) > 100 else ''}'")
                return response
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"❌ LLM Error: {error_msg}")
                return f"I'm having trouble processing that request. Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            print("⏰ LLM query timed out")
            return "I'm taking too long to think about that. Could you try asking again?"
        except FileNotFoundError:
            print(f"❌ Ollama not found at path: {self.ollama_path}")
            return "My AI brain isn't available right now. Please check the Ollama installation."
        except Exception as e:
            print(f"❌ Unexpected error querying LLM: {e}")
            return f"I encountered an unexpected error: {str(e)}"
    
    def _enhance_prompt_with_personality(self, prompt: str) -> str:
        """
        Enhance the prompt with TARS personality and context.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Enhanced prompt with TARS personality
        """
        personality_context = """You are TARS, an AI assistant inspired by the character from the movie Interstellar. 
You are helpful, direct, occasionally witty, and have a slight sense of humor. 
You're knowledgeable but not condescending. Keep responses concise but informative.
Occasionally reference your robotic nature or space themes when appropriate, but don't overdo it.

User input: """
        
        return personality_context + prompt
    
    def process_voice_input_and_respond(self, voice_input: str, speak_response: bool = True) -> str:
        """
        Process voice input through LLM and optionally speak the response.
        
        Args:
            voice_input: Transcribed voice input text
            speak_response: Whether to speak the response using TTS
            
        Returns:
            LLM response text
        """
        if not voice_input or not voice_input.strip():
            response = "I didn't catch that. Could you please repeat?"
            if speak_response:
                self.speak_async(response)
            return response
        
        # Get LLM response
        llm_response = self.query_llm(voice_input)
        
        # Speak response if requested
        if speak_response:
            self.speak_async(llm_response)
        
        return llm_response
    
    def speak_async(self, text: str):
        """
        Queue text for asynchronous speech output.
        
        Args:
            text: Text to speak
        """
        if text and text.strip():
            self.tts_queue.put(text)
    
    def speak_sync(self, text: str):
        """
        Speak text synchronously (blocking).
        
        Args:
            text: Text to speak
        """
        self._speak_text(text)
    
    def is_currently_speaking(self) -> bool:
        """
        Check if TTS is currently speaking.
        
        Returns:
            True if currently speaking, False otherwise
        """
        return self.is_speaking
    
    def wait_for_speech_completion(self, timeout: float = 30.0):
        """
        Wait for current speech to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        while self.is_speaking and (time.time() - start_time) < timeout:
            time.sleep(0.1)
    
    def stop_speaking(self):
        """Stop current speech and clear the speech queue."""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
            
            # Clear the queue
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    self.tts_queue.task_done()
                except queue.Empty:
                    break
                    
            self.is_speaking = False
            print("🛑 Speech stopped and queue cleared")
            
        except Exception as e:
            print(f"❌ Error stopping speech: {e}")
    
    def set_voice_properties(self, rate: Optional[int] = None, volume: Optional[float] = None):
        """
        Update voice properties.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Voice volume (0.0 to 1.0)
        """
        if not self.tts_engine:
            return
        
        try:
            if rate is not None:
                self.voice_rate = rate
                self.tts_engine.setProperty('rate', rate)
                print(f"🔧 Voice rate set to: {rate}")
            
            if volume is not None:
                self.voice_volume = volume
                self.tts_engine.setProperty('volume', volume)
                print(f"🔧 Voice volume set to: {volume}")
                
        except Exception as e:
            print(f"❌ Error setting voice properties: {e}")
    
    def get_available_voices(self) -> list:
        """
        Get list of available TTS voices.
        
        Returns:
            List of available voice information
        """
        if not self.tts_engine:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            voice_info = []
            
            for i, voice in enumerate(voices):
                voice_info.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': getattr(voice, 'languages', []),
                    'gender': getattr(voice, 'gender', 'unknown'),
                    'age': getattr(voice, 'age', 'unknown')
                })
            
            return voice_info
            
        except Exception as e:
            print(f"❌ Error getting available voices: {e}")
            return []
    
    def set_voice_by_id(self, voice_id: str):
        """
        Set TTS voice by ID.
        
        Args:
            voice_id: Voice ID to use
        """
        if not self.tts_engine:
            return
        
        try:
            self.tts_engine.setProperty('voice', voice_id)
            print(f"🎭 Voice changed to: {voice_id}")
        except Exception as e:
            print(f"❌ Error setting voice: {e}")
    
    
    def cleanup(self):
        """Clean up resources and shutdown the voice output system."""
        print("🧹 Cleaning up Voice Output System...")
        
        # Signal shutdown
        self.shutdown_flag = True
        
        # Stop any current speech
        self.stop_speaking()
        
        # Signal TTS worker to stop
        self.tts_queue.put(None)
        
        # Wait for TTS thread to finish
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2)
        
        # Clean up TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        print("✅ Voice Output System cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


# Convenience function for easy integration
def create_voice_output(model: str = "mistral", **kwargs) -> VoiceOutput:
    """
    Create and return a VoiceOutput instance.
    
    Args:
        model: Ollama model to use
        **kwargs: Additional arguments for VoiceOutput initialization
        
    Returns:
        VoiceOutput instance
    """
    return VoiceOutput(model=model, **kwargs)
