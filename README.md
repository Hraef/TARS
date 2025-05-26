# TARS
This project is inspired by TARS from Interstellar. Ideally this would be able to run with or without internet connectivity. Below is my thoughts on completing this to be ran locally with no internet and is subject to change.

## Ideas
- Run this with the ability to connect to the internet if needed
- Run tasks/jobs when prompted through speech
- Have the ability to connect to IoT devices for advanced AI capabilites
- Build TARS to his complete full pristine perfect self.... while keeping the robots from enslaving humans at bay


## Tools
- Whisper
  - can be used offline 
```import whisper```

  ```model = whisper.load_model("base")```
  - https://github.com/openai/whisper
- Chat enginer (AI)
  - llama.cpp or Mistral (currently leaning to Mistral)
  - https://ollama.com/library/mistral
- Text-to-Speech
  - pyttsx3
  - https://pypi.org/project/pyttsx3/
