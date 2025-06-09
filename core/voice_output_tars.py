#!/usr/bin/env python3
"""
TARS Voice Output with Authentic Voice Cloning
This module implements multiple methods to achieve the actual TARS voice from Interstellar.
"""

import os
import sys
import subprocess
import threading
import queue
import time
import tempfile
import wave
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class TARSVoiceOutput:
    """Advanced TARS voice output with multiple voice synthesis methods."""
    
    def __init__(self, model: str = "mistral", voice_method: str = "enhanced_espeak"):
        """
        Initialize TARS voice output with multiple synthesis options.
        
        Args:
            model: Ollama model for LLM processing
            voice_method: Voice synthesis method:
                - "enhanced_espeak": Enhanced eSpeak with TARS effects
                - "coqui_tts": Coqui TTS with voice cloning
                - "tortoise_tts": Tortoise TTS for high quality
                - "hybrid": Combination of methods
        """
        print("🤖 Initializing Authentic TARS Voice System...")
        
        # LLM Configuration
        self.model = model
        self.ollama_path = os.getenv("OLLAMA_PATH", "ollama")
        
        # Voice Configuration
        self.voice_method = voice_method
        self.is_speaking = False
        self.tts_queue = queue.Queue()
        self.shutdown_flag = False
        
        # Initialize voice system based on method
        self._init_voice_system()
        
        # Start TTS worker thread
        self._start_tts_worker()
        
        print("✅ Authentic TARS Voice System initialized!")
    
    def _init_voice_system(self):
        """Initialize the appropriate voice synthesis system."""
        try:
            if self.voice_method == "enhanced_espeak":
                self._init_enhanced_espeak()
            elif self.voice_method == "coqui_tts":
                self._init_coqui_tts()
            elif self.voice_method == "tortoise_tts":
                self._init_tortoise_tts()
            elif self.voice_method == "hybrid":
                self._init_hybrid_system()
            else:
                print(f"⚠️  Unknown voice method: {self.voice_method}, falling back to enhanced eSpeak")
                self._init_enhanced_espeak()
                
        except Exception as e:
            print(f"❌ Error initializing voice system: {e}")
            print("🔄 Falling back to basic eSpeak...")
            self._init_enhanced_espeak()
    
    def _init_enhanced_espeak(self):
        """Initialize enhanced eSpeak with TARS-optimized settings."""
        print("🔊 Initializing Enhanced eSpeak for TARS...")
        
        # TARS-optimized eSpeak parameters (closest to movie voice)
        self.espeak_params = [
            'espeak',
            '-s', '155',     # Speed: Deliberate pace like TARS
            '-p', '12',      # Pitch: Lower for authority
            '-a', '100',     # Amplitude: Full volume
            '-g', '10',      # Gap: Slight pauses between words
            '-v', 'en+m3',   # Voice: Male voice variant 3
            '-k', '5',       # Emphasis on certain syllables
        ]
        
        # Test eSpeak availability
        try:
            subprocess.run(['espeak', '--version'], capture_output=True, check=True)
            print("✅ Enhanced eSpeak initialized")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ eSpeak not found - please install: sudo apt install espeak")
            raise
    
    def _init_coqui_tts(self):
        """Initialize Coqui TTS for voice cloning."""
        print("🔊 Initializing Coqui TTS for TARS voice cloning...")
        
        try:
            from TTS.api import TTS
            
            # Try to load a suitable voice model
            # You can train your own TARS model or use a close approximation
            model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Base model
            
            self.coqui_tts = TTS(model_name=model_name)
            print("✅ Coqui TTS initialized")
            
            # Note: For authentic TARS voice, you would need to:
            # 1. Collect TARS audio samples from the movie
            # 2. Train a custom model or use voice conversion
            # 3. This is a starting point for voice cloning
            
        except ImportError:
            print("❌ Coqui TTS not installed - run: pip install TTS")
            raise
        except Exception as e:
            print(f"❌ Error initializing Coqui TTS: {e}")
            raise
    
    def _init_tortoise_tts(self):
        """Initialize Tortoise TTS for high-quality voice synthesis."""
        print("🔊 Initializing Tortoise TTS...")
        
        try:
            # Tortoise TTS would require additional setup
            # This is a placeholder for the implementation
            print("⚠️  Tortoise TTS requires additional setup - see documentation")
            print("🔄 Falling back to enhanced eSpeak...")
            self._init_enhanced_espeak()
            
        except Exception as e:
            print(f"❌ Error initializing Tortoise TTS: {e}")
            self._init_enhanced_espeak()
    
    def _init_hybrid_system(self):
        """Initialize hybrid system with multiple voice methods."""
        print("🔊 Initializing Hybrid TARS Voice System...")
        
        # Try to initialize multiple systems
        self.available_methods = []
        
        try:
            self._init_enhanced_espeak()
            self.available_methods.append("enhanced_espeak")
        except:
            pass
        
        try:
            self._init_coqui_tts()
            self.available_methods.append("coqui_tts")
        except:
            pass
        
        if self.available_methods:
            print(f"✅ Hybrid system initialized with: {', '.join(self.available_methods)}")
        else:
            raise Exception("No voice synthesis methods available")
    
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
                
                # Speak the text using appropriate method
                self._speak_text(text)
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error in TTS worker: {e}")
    
    def _speak_text(self, text: str):
        """Speak text using the configured voice method."""
        if not text.strip():
            return
        
        try:
            self.is_speaking = True
            print(f"🤖 TARS speaking: {text}")
            
            if self.voice_method == "enhanced_espeak" or hasattr(self, 'espeak_params'):
                self._speak_with_enhanced_espeak(text)
            elif self.voice_method == "coqui_tts" and hasattr(self, 'coqui_tts'):
                self._speak_with_coqui_tts(text)
            elif self.voice_method == "hybrid":
                self._speak_with_hybrid(text)
            else:
                # Fallback to basic eSpeak
                subprocess.run(['espeak', text], capture_output=True, timeout=10)
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            print(f"💬 TARS says (text-only): {text}")
        finally:
            self.is_speaking = False
    
    def _speak_with_enhanced_espeak(self, text: str):
        """Speak using enhanced eSpeak with TARS optimizations."""
        # Apply TARS-specific text preprocessing
        processed_text = self._preprocess_text_for_tars(text)
        
        # Use the optimized eSpeak parameters
        cmd = self.espeak_params + [processed_text]
        subprocess.run(cmd, capture_output=True, timeout=15)
    
    def _speak_with_coqui_tts(self, text: str):
        """Speak using Coqui TTS with voice cloning."""
        try:
            # Generate audio with Coqui TTS
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate speech
            self.coqui_tts.tts_to_file(text=text, file_path=temp_path)
            
            # Play the audio file
            self._play_audio_file(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"❌ Coqui TTS error: {e}")
            # Fallback to eSpeak
            self._speak_with_enhanced_espeak(text)
    
    def _speak_with_hybrid(self, text: str):
        """Speak using hybrid approach - best available method."""
        if "coqui_tts" in self.available_methods:
            self._speak_with_coqui_tts(text)
        elif "enhanced_espeak" in self.available_methods:
            self._speak_with_enhanced_espeak(text)
        else:
            subprocess.run(['espeak', text], capture_output=True, timeout=10)
    
    def _preprocess_text_for_tars(self, text: str) -> str:
        """Preprocess text to make it sound more like TARS."""
        # TARS speaks in a very measured, precise way
        processed = text
        
        # Add slight pauses for dramatic effect (using eSpeak markup)
        processed = processed.replace('.', '[[slnc 200]].')  # Pause after sentences
        processed = processed.replace(',', '[[slnc 100]],')  # Brief pause after commas
        processed = processed.replace(':', '[[slnc 150]]:')  # Pause after colons
        
        # Emphasize certain TARS-like phrases
        tars_phrases = {
            'probability': '[[emph on]]probability[[emph off]]',
            'percent': '[[emph on]]percent[[emph off]]',
            'systems': '[[emph on]]systems[[emph off]]',
            'operational': '[[emph on]]operational[[emph off]]',
            'affirmative': '[[emph on]]affirmative[[emph off]]',
            'negative': '[[emph on]]negative[[emph off]]',
        }
        
        for phrase, replacement in tars_phrases.items():
            processed = processed.replace(phrase, replacement)
        
        return processed
    
    def _play_audio_file(self, audio_path: str):
        """Play an audio file using available system tools."""
        try:
            # Try different audio players
            players = ['aplay', 'paplay', 'ffplay', 'mpv']
            
            for player in players:
                try:
                    subprocess.run([player, audio_path], 
                                 capture_output=True, 
                                 timeout=30, 
                                 check=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            print("⚠️  No audio player found - audio file generated but not played")
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    def query_llm(self, prompt: str, timeout: int = 60) -> str:
        """Query the local LLM with enhanced TARS personality."""
        try:
            print(f"🧠 TARS analyzing: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Enhanced TARS personality prompt
            enhanced_prompt = self._enhance_prompt_with_tars_personality(prompt)
            
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
                print(f"✅ TARS response generated")
                return response
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return f"TARS AI systems experiencing difficulty. Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "TARS processing timeout. Please repeat the query."
        except Exception as e:
            return f"TARS AI interface error: {str(e)}"
    
    def _enhance_prompt_with_tars_personality(self, prompt: str) -> str:
        """Enhance prompt with authentic TARS personality."""
        personality_context = """You are TARS from the movie Interstellar. You are:

- Highly intelligent, logical, and precise
- Speak in a measured, authoritative tone
- Occasionally use percentages and probabilities
- Have a dry sense of humor when appropriate
- Direct and honest, sometimes bluntly so
- Loyal and helpful to your human companions
- Reference your robotic nature and capabilities when relevant
- Use phrases like "affirmative", "negative", "systems operational"
- Sometimes mention probability calculations or system status

Stay in character as TARS. Keep responses concise but informative.

Human input: """
        
        return personality_context + prompt
    
    def process_voice_input_and_respond(self, voice_input: str, speak_response: bool = True) -> str:
        """Process voice input and respond as TARS."""
        if not voice_input or not voice_input.strip():
            response = "TARS audio input not detected. Please repeat your transmission."
            if speak_response:
                self.speak_async(response)
            return response
        
        # Get TARS response
        tars_response = self.query_llm(voice_input)
        
        # Speak response if requested
        if speak_response:
            self.speak_async(tars_response)
        
        return tars_response
    
    def speak_async(self, text: str):
        """Queue text for asynchronous speech output."""
        if text and text.strip():
            self.tts_queue.put(text)
    
    def speak_sync(self, text: str):
        """Speak text synchronously (blocking)."""
        self._speak_text(text)
    
    def is_currently_speaking(self) -> bool:
        """Check if TARS is currently speaking."""
        return self.is_speaking
    
    def wait_for_speech_completion(self, timeout: float = 30.0):
        """Wait for current speech to complete."""
        start_time = time.time()
        while self.is_speaking and (time.time() - start_time) < timeout:
            time.sleep(0.1)
    
    def test_tars_voice(self):
        """Test the TARS voice system with movie-like phrases."""
        test_phrases = [
            "TARS voice systems online and operational.",
            "Probability of successful voice synthesis: ninety-seven percent.",
            "All systems nominal. Standing by for orders.",
            "Affirmative. TARS is ready to assist.",
            "Humor setting currently at seventy-five percent.",
            "That's impossible. Let me recalculate... still impossible."
        ]
        
        print("🎭 Testing TARS voice with authentic phrases...")
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n🤖 Test {i}: {phrase}")
            self.speak_sync(phrase)
            time.sleep(1)  # Brief pause between tests
        
        print("\n✅ TARS voice test complete!")
    
    def cleanup(self):
        """Clean up TARS voice system resources."""
        print("🧹 TARS shutting down voice systems...")
        
        self.shutdown_flag = True
        
        # Clear speech queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
        
        # Signal TTS worker to stop
        self.tts_queue.put(None)
        
        # Wait for TTS thread to finish
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2)
        
        print("✅ TARS voice systems offline")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


# Convenience function
def create_tars_voice(voice_method: str = "enhanced_espeak", **kwargs) -> TARSVoiceOutput:
    """
    Create and return a TARSVoiceOutput instance.
    
    Args:
        voice_method: Voice synthesis method to use
        **kwargs: Additional arguments
        
    Returns:
        TARSVoiceOutput instance
    """
    return TARSVoiceOutput(voice_method=voice_method, **kwargs) 