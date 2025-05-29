# TARS
This project is inspired by TARS from Interstellar. Ideally this would be able to run with or without internet connectivity. Below is my thoughts on completing this to be ran locally with no internet and is subject to change.

## Current Status - Phase 1 Complete! 🎉

✅ **Voice Input System**: Basic voice recording and Whisper transcription  
✅ **Interactive Mode**: Continuous voice interaction with TARS  
✅ **Basic Commands**: Hello, time, date, help, quit functionality  
🔄 **Next**: AI engine integration (Ollama/Mistral)  

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test voice input:**
   ```bash
   python test_voice.py
   ```

3. **Run TARS:**
   ```bash
   python main.py
   ```
   Or with a specific Whisper model:
   ```bash
   python main.py tiny    # Fastest
   python main.py base    # Default, good balance
   python main.py small   # Better accuracy
   ```

   **Alternative: Use the provided shell script (recommended for cleaner output):**
   ```bash
   ./run_tars.sh          # Runs with audio error suppression
   ./run_tars.sh base     # With specific Whisper model
   ```
   *Note: Make the script executable first with `chmod +x run_tars.sh`*

## Usage

### Interactive Mode
- Run `python main.py` to start TARS
- Speak your command when prompted
- Press Enter to stop recording
- TARS will transcribe and respond
- Say "quit", "exit", or "goodbye" to stop

### Current Commands
- **"Hello"** - Greeting
- **"What time is it?"** - Current time
- **"What's the date?"** - Current date  
- **"Help"** - List available commands
- **"Quit"** - Exit TARS

## Ideas
- Run this with the ability to connect to the internet if needed
- Run tasks/jobs when prompted through speech
- Have the ability to connect to IoT devices for advanced AI capabilites
- Build TARS to his complete full pristine perfect self.... while keeping the robots from enslaving humans at bay


## Tools
- **Whisper** ✅ IMPLEMENTED
  - can be used offline
  - https://github.com/openai/whisper
  ```Python
  import whisper 
  model = whisper.load_model("base")
  ```
- **Chat engine (AI)** 🔄 NEXT PHASE
  - llama.cpp or Mistral (currently leaning to Mistral)
  - https://ollama.com/library/mistral
- **Text-to-Speech** 📋 PLANNED
  - pyttsx3
  - https://pypi.org/project/pyttsx3/

## Architecture

```
TARS/
├── main.py              # Main application entry point
├── core/
│   ├── voice_input.py   # ✅ Voice recording & Whisper transcription
│   ├── voice_output.py  # 📋 Text-to-speech (planned)
│   └── engine.py        # 🔄 AI processing (next phase)
├── test_voice.py        # Voice input testing
└── requirements.txt     # Dependencies
```

## Troubleshooting

### Common Issues
- **No microphone detected**: Check microphone connection and permissions
- **PyAudio errors**: May need system audio libraries:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install portaudio19-dev
  
  # macOS
  brew install portaudio
  ```
- **Whisper model download**: First run downloads models (~74MB for base)
- **Permission errors**: Ensure microphone access is granted

### Performance Tips
- Use `tiny` model for fastest response (less accurate)
- Use `base` model for good balance (recommended)
- Use `small`+ models for better accuracy (slower)
