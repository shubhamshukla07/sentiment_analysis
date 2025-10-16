import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ----------- Text model training & loading -----------
@st.cache_data(show_spinner=True)
def load_and_train_text_model():
    df = pd.read_csv("C:/Users/shubh/OneDrive/Desktop/sentiment/dataset.csv")
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=['Tweets', 'Labels'])
    df = df[df['Tweets'].str.strip() != ""]
    df = df.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['Tweets'], df['Labels'], test_size=0.2, random_state=42, stratify=df['Labels']
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Optionally save
    joblib.dump(clf, 'logreg_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    return clf, vectorizer, accuracy, report

# ----------- Face emotion model loading -----------
@st.cache_resource(show_spinner=True)
def load_emotion_model():
    model_path = r"C:\Users\shubh\OneDrive\Desktop\sentiment\fer2013_mini_XCEPTION.102-0.66.hdf5"
    model = load_model(model_path, compile=False)
    return model

# Preprocessing face for emotion model
def preprocess_face(gray_frame, x, y, w, h):
    face = gray_frame[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

# ------------ Main Streamlit app ----------------
st.title("🧠 Depression & Emotion Detection")

# Load models
clf, vectorizer, accuracy, report = load_and_train_text_model()
emotion_model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Sidebar for text sentiment detection
st.sidebar.header("Text-based Depression Sentiment")
age = st.sidebar.slider("Select your age:", 0, 120, 30)
user_input = st.sidebar.text_area("Enter text here:", height=150)

label_meanings = {
    "no": "No depression detected / Neutral sentiment",
    "postpartum": "Postpartum depression",
    "atypical": "Atypical depression",
    "bipolar": "Bipolar disorder",
    "major depressive": "Major depressive disorder",
    "psychotic": "Psychotic depression"
}

st.sidebar.markdown("### Label meanings:")
for label, meaning in label_meanings.items():
    st.sidebar.write(f"**{label}**: {meaning}")

st.sidebar.write(f"### Model Accuracy on test set: {accuracy:.2%}")

with st.sidebar.expander("View Classification Report"):
    st.text(report)

if st.sidebar.button("Predict"):
    if user_input.strip() == "":
        st.sidebar.warning("Please enter some text to predict.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = clf.predict(input_vec)[0]
        st.sidebar.success(f"Predicted Sentiment: **{prediction}**")

# ---------------- Webcam emotion detection ----------------
st.markdown("### Real-time Face Emotion Detection")

# Streamlit provides a special video input widget: `st.camera_input`
# We'll use OpenCV with Streamlit's webcam streaming by capturing frames manually

# Create a placeholder for the video frame
video_placeholder = st.empty()

# Function to run webcam and predict emotion
def run_webcam_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = preprocess_face(gray, x, y, w, h)
                preds = emotion_model.predict(face)
                emotion = emotion_labels[np.argmax(preds)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 0, 0), 2)

            # Resize frame smaller for sidebar display
            frame = cv2.resize(frame, (320, 240))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame)

    finally:
        cap.release()

# Run webcam only when user clicks button
if st.button("Start Webcam Emotion Detection"):
    run_webcam_emotion()
else:
    st.info("Click the button above to start the webcam for face emotion detection.")

