import cv2
import pyttsx3
import PySimpleGUI as sg
import speech_recognition as sr
from deepface import DeepFace
from sklearn.naive_bayes import GaussianNB
import random
import json
import datetime

# Module 1: Emotion Detection
def detect_emotion_from_face(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except:
        return "neutral"
    
def detect_emotion_from_voice():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Speak briefly for voice emotion analysis...")
            audio = recognizer.listen(source, timeout=5)
        # Placeholder: In real system, run sudio through emotion model
        return random.choice(["happy", "sad", "neutral", "angry"])
    except:
        return "neutral"
    
# Module 2: Adaptive Learning Engine
class AdaptiveTutor:
    def __init__(self):
        self.tutor_voice = pyttsx3.init()
        self.model = GaussianNB()
        self.data =[]
        self.labels = []

    def speak(self, text):
        print("Tutor:", text)
        self.tutor_voice.say(text)
        self.tutor_voice.runAndWait()

    def adapt_content(self, emotion, performance):
        if emotion == "happy" and performance > 80:
            return "Advance to next level"
        elif emotion in ["sad", "angry"]:
            return "Slow down, offer help"
        else:
            return "Keep steady pace"
        
    def train_model(self):
        if self.data and self.labels:
            self.model.fit(self.data, self.labels)

    def predict_strategy(self, emotion, score):
        emotion_code = {"happy": 0, "sad": 1, "angry": 2, "neutral": 3}
        input_data = [[emotion_code.get(emotion, 3), score]]
        return self.model.predict(input_data)[0]
    
# Module 3: Simulated Learning Environment
def simulate_vr_environment(topic):
    print(f"Entering VR Environment: {topic}")
    # In real VR, this would be Unity/Unreal - placeholder text only
    print(f"Showing immersive {topic} simulation...")

# Module 4: Session Feedback and Analytics
def save_session_analytics(student_name, emotion, score, strategy):
    session_data = {
        "student": student_name,
        "datetime": str(datetime.datetime.now()),
        "emotion": emotion,
        "score": score,
        "staregy": strategy
    }
    with open("session_log.json", "a") as file:
        file.write(json.dumps(session_data) + "\n")

# Main Flow
def run_system():
    tutor = AdaptiveTutor()
    sg.theme('DarkBlue')

    layout = [
        [sg.Text('Enter Student Name:'), sg.InputText()],
        [sg.Button('Start Learning Session'), sg.Button('Exit')]
    ]

    window = sg.Window('AI VR Tutor', layout)

    while True:
        event, values = window.read()
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        if event == 'Start Learning Session':
            student = values[0]
            simulate_vr_environment("Ancient Egypt")

            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            emotion_face = detect_emotion_from_face(frame)
            emotion_voice = detect_emotion_from_voice()
            emotion = emotion_voice if emotion_voice != "neutral" else emotion_face
            cam.release()

            score = random.randint(50, 100) # Simulate quiz score
            strategy = tutor.adapt_content(emotion, score)
            tutor.speak(f"Welcome {student}, I noticed you're feeling {emotion}. Based on your performance, we will {strategy.lower()}.")

            save_session_analytics(student, emotion, score, strategy)

    window.close()

# Run the system
if __name__ == "__main__":
    run_system()
