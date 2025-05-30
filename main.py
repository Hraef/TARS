#!/usr/bin/env python3
"""
TARS - AI Assistant inspired by Interstellar
Main application file that orchestrates voice input, AI processing, and voice output.
"""
from core.engine import query_local_llm
import sys
import os

# Suppress ALSA and JACK audio error messages
os.environ['ALSA_PCM_CARD'] = 'default'
os.environ['ALSA_PCM_DEVICE'] = '0'
os.environ['ALSA_MIXER_CARD'] = 'default'
os.environ['SDL_AUDIODRIVER'] = 'pulse'
os.environ['PULSE_RUNTIME_PATH'] = '/dev/null'

# Redirect stderr to suppress audio errors during imports
import contextlib
import io

def suppress_audio_errors():
    """Context manager to suppress audio initialization errors."""
    return contextlib.redirect_stderr(io.StringIO())

# Use the suppression when importing audio-related modules
with suppress_audio_errors():
    from core.voice_input import VoiceInput


class TARS:
    """Main TARS AI Assistant class."""
    
    def __init__(self, whisper_model: str = "base"):
        """
        Initialize TARS with voice capabilities.
        
        Args:
            whisper_model: Whisper model size to use for voice recognition
        """
        print("🤖 Initializing TARS...")
        self.voice_input = VoiceInput(model_size=whisper_model)
        print("✅ TARS is ready!")
    
    def listen_and_respond(self):
        """Listen for voice input and provide a basic response."""
        try:
            print("\n" + "="*50)
            print("🎤 TARS is listening...")
            print("Speak your command, then press Enter to stop recording.")
            print("="*50)
            
            # Get voice input
            user_input = self.voice_input.record_and_transcribe()
            
            if user_input:
                print(f"\n🧠 TARS heard: '{user_input}'")
                
                # Basic response logic (Phase 1 - simple responses)
                response = self.process_command(user_input)
                print(f"🤖 TARS responds: {response}")
                
                return user_input, response
            else:
                print("❌ No speech detected or transcription failed.")
                return None, None
                
        except Exception as e:
            print(f"❌ Error during voice processing: {e}")
            return None, None
    
    def process_command(self, command: str) -> str:
        """
        Process user command and generate response.
        This is a placeholder for Phase 1 - will be replaced with AI engine later.
        
        Args:
            command: Transcribed user command
            
        Returns:
            Response string
        """
        command_lower = command.lower().strip()

        try:
            response = self.engine.chat(command)
            if response:
                return response
        except Exception as e:
            print(f"⚠️  Error querying AI engine: {e}. Reverting to basic responses.")
        
        # Basic command responses for Phase 1
        if "hello" in command_lower or "hi" in command_lower:
            return "Hello! I'm TARS, your AI assistant. How can I help you today?"
        
        elif "time" in command_lower:
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."
        
        elif "date" in command_lower:
            from datetime import datetime
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}."
        
        elif "weather" in command_lower:
            return "I don't have weather data access yet, but I'm working on it!"
        
        elif "quit" in command_lower or "exit" in command_lower or "goodbye" in command_lower:
            return "Goodbye! TARS shutting down."
        
        elif "help" in command_lower:
            return ("I can respond to: hello, time, date, weather (coming soon), "
                   "help, or quit. More capabilities coming soon!")
        
        else:
            return (f"I heard you say '{command}', but I don't know how to respond to that yet. "
                   "Try saying 'help' to see what I can do!")
    
    def run_interactive_mode(self):
        """Run TARS in interactive mode with continuous voice input."""
        print("\n🚀 Starting TARS Interactive Mode")
        print("Say 'quit', 'exit', or 'goodbye' to stop TARS")
        
        try:
            while True:
                user_input, response = self.listen_and_respond()
                
                if user_input and any(word in user_input.lower() 
                                    for word in ["quit", "exit", "goodbye"]):
                    break
                
                print("\n" + "-"*30)
                print("Ready for next command...")
                
        except KeyboardInterrupt:
            print("\n\n🛑 TARS interrupted by user.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up TARS resources."""
        print("\n🧹 Cleaning up TARS...")
        if hasattr(self, 'voice_input'):
            self.voice_input.cleanup()
        print("✅ TARS shutdown complete.")


def main():
    """Main entry point for TARS application."""
    print("🌟 Welcome to TARS - Your AI Assistant")
    print("Inspired by the AI from Interstellar")
    
    # Parse command line arguments for model selection
    whisper_model = "base"  # Default model
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        if model_arg in ["tiny", "base", "small", "medium", "large"]:
            whisper_model = model_arg
        else:
            print(f"⚠️  Unknown model '{model_arg}', using 'base' instead.")
    
    print(f"🔧 Using Whisper model: {whisper_model}")
    
    try:
        # Initialize and run TARS
        tars = TARS(whisper_model=whisper_model)
        tars.run_interactive_mode()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()