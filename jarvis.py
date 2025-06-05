import os
import json
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio


os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import pipeline


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

running = True

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_command(text):
    labels = [
    "open_browser", "play_music", "shutdown", "exit", 
    "open_notepad", "open_code", "open_folder", "open_drive"
]
    if not text.strip():
        return "unknown"

    result = classifier(text, candidate_labels=labels)
    command = result['labels'][0]
    confidence = result['scores'][0]
    return command if confidence > 0.5 else "unknown"



def open_browser():
    speak("Opening Chrome")
    os.system("start chrome")

def play_music():
    speak("Playing music")
    os.system("start Music")  

def shutdown():
    speak("Shutting down the system")
    os.system("shutdown /s /t 1")

def exit_jarvis():
    global running
    speak("Goodbye")
    running = False
def open_notepad():
    speak("Opening Notepad")
    os.system("notepad")

def open_code():
    speak("Opening VS Code")
    os.system("code")  

def open_folder():
    speak("Opening your projects folder")
    os.system(r'start "" "C:\offline ai"')  

def open_drive():
    speak("Opening D drive")
    os.system("start D:\\")

def unknown():
    speak("I didn't understand that. Please try again.")



command_actions = {
    "open_browser": open_browser,
    "play_music": play_music,
    "shutdown": shutdown,
    "exit": exit_jarvis,
    "open_notepad": open_notepad,
    "open_code": open_code,
    "open_folder": open_folder,
    "open_drive": open_drive,
    "unknown": unknown
}



def run_assistant():
    global running
    running = True

    model = Model("models/vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    speak("Assistant is now listening")

    while running:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").lower()
            print("Heard:", text)

            command = get_command(text)
            print("Understood as:", command)

            action = command_actions.get(command, unknown)
            action()

    stream.stop_stream()
    stream.close()
    mic.terminate()
    print("Assistant stopped listening")

def stop_assistant():
    global running
    running = False
