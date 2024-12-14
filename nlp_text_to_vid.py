import spacy
import speech_recognition as sr
import os
import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
from IPython.display import display, Video

class TextToISL:
    def __init__(self, dataset_path):
        self.nlp = spacy.load("en_core_web_md")
        self.recognizer = sr.Recognizer()
        self.dataset_path = dataset_path
        self.words = self.load_word_folders()

    def load_word_folders(self):
        """Load all word folders in the ISL directory."""
        isl_path = os.path.join(self.dataset_path, "isl")
        if not os.path.isdir(isl_path):
            raise FileNotFoundError(f"'ISL' folder not found in {self.dataset_path}")

        words = [folder.lower() for folder in os.listdir(isl_path) if os.path.isdir(os.path.join(isl_path, folder))]
        if not words:
            raise ValueError(f"No word folders found in {isl_path}")
        return set(words)  # Using a set for quick lookups

    def get_from_user(self):
        """Streamlit interface for selecting text or audio."""
        choice = st.selectbox("Choose Input Method", ("Text", "Audio"))

        if choice == "Text":
            inp_sent = st.text_input("Enter the text")
            return inp_sent
        elif choice == "Audio":
            audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])
            if audio_file:
                try:
                    with sr.AudioFile(audio_file) as source:
                        audio = self.recognizer.record(source)
                    inp_sent = self.recognizer.recognize_google(audio)
                    st.write(f"You said: {inp_sent}")
                    return inp_sent
                except sr.UnknownValueError:
                    st.error("Sorry, I couldn't understand the audio.")
                    return None
                except sr.RequestError:
                    st.error("Speech Recognition service is unavailable.")
                    return None
        return None

    def lower_case(self, text):
        return text.lower()

    def tokenize(self, text):
        doc = self.nlp(text)
        return [(i, word.text) for i, word in enumerate(doc)]

    def lemmatize(self, text):
        doc = self.nlp(text)
        return [word.lemma_ for word in doc if not word.is_stop]

    def POS(self, text):
        doc = self.nlp(text)
        return [(word.text, word.pos_, word.dep_) for word in doc]

    def convert_to_isl(self, text):
        """Convert the text to ISL format based on grammatical rules."""
        doc = self.nlp(text)

        subject = ""
        subject_adjectives = []
        verb = ""
        objects = []
        object_adjectives = []
        possessive = ""
        adverbs = []
        punctuation = ""

        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text
                # Collect adjectives modifying the subject
                for child in token.children:
                    if child.dep_ == "amod":
                        subject_adjectives.append(child.text)

            if token.pos_ == "VERB" and not verb:
                verb = token.text

            if token.dep_ in ("dobj", "pobj", "attr", "obj"):
                objects.append(token.text)
                # Collect adjectives modifying the object
                for child in token.children:
                    if child.dep_ == "amod":
                        object_adjectives.append(child.text)

            if token.tag_ in ("WP", "WRB"):
                question_word = token.text

            if token.dep_ == "poss":
                possessive = token.text

            if token.pos_ == "ADV":
                adverbs.append(token.text)

            if token.pos_ == "PUNCT":
                punctuation = token.text

        # Construct the ISL sentence
        isl_sentence = []

        if possessive:
            isl_sentence.append(possessive)

        if subject:
            isl_sentence.append(subject)

        if subject_adjectives:
            isl_sentence.extend(subject_adjectives)

        if objects:
            if object_adjectives:
                isl_sentence.extend(object_adjectives)
            isl_sentence.extend(objects)

        if adverbs:
            isl_sentence.extend(adverbs)

        if verb:
            isl_sentence.append(verb)

        if punctuation:
            isl_sentence.append(punctuation)

        return self.map_to_dataset(" ".join(isl_sentence))

    def map_to_dataset(self, sentence):
        """Map the final ISL sentence words to dataset folders."""
        mapped_sentence = []

        for word in sentence.split():
            if word in self.words:
                mapped_sentence.append(word)
            else:
                st.warning(f"Word '{word}' not found in dataset, skipping.")

        return " ".join(mapped_sentence)

    def process_text(self):
        inp_sent = self.get_from_user()

        if inp_sent:
            lowercased_text = self.lower_case(inp_sent)
            st.write(f"Lowercased Text: {lowercased_text}")
            st.write("Tokens:", self.tokenize(lowercased_text))
            st.write("POS Tags:", self.POS(lowercased_text))
            isl_sentence = self.convert_to_isl(lowercased_text)
            st.write(f"Mapped ISL Sentence: {isl_sentence}")

            self.concatenate_and_display_videos(isl_sentence)

    def concatenate_and_display_videos(self, isl_sentence):
        words = isl_sentence.split()
        video_clips = []

        # Base path for ISL video dataset
        base_path = os.path.join(self.dataset_path, "isl")

        for word in words:
            word_folder = os.path.join(base_path, word)
            if os.path.isdir(word_folder):
                # Load the first video in the folder
                video_files = [f for f in os.listdir(word_folder) if f.endswith(('.mp4', '.avi','.mov'))]
                if video_files:
                    video_path = os.path.join(word_folder, video_files[0])
                    try:
                        video_clip = VideoFileClip(video_path)
                        video_clips.append(video_clip)
                    except Exception as e:
                        st.error(f"Error loading video for '{word}': {e}")
            else:
                st.warning(f"No folder found for word: {word}")

        if video_clips:
            # Concatenate video clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            # Save the final video to disk
            output_path = "/tmp/final_isl_video.mp4"  # Temporary path for Streamlit
            final_video.write_videofile(output_path, codec="libx264", audio=False)

            # Check video duration for debugging
            st.write(f"Video Duration: {final_video.duration} seconds")

            # Display video in Streamlit
            st.video(output_path)  # This will display the video inline in Streamlit
        else:
            st.error("No videos found to concatenate for the given ISL sentence.")

if __name__ == "__main__":
    dataset_path = "/Users/shriya/Documents/GitHub/isl_neutrino/isl_sch 2"  # Replace with your dataset's path
    text_to_ISL = TextToISL(dataset_path)
    text_to_ISL.process_text()
