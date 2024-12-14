import spacy
import pandas as pd

class TextToISL:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def process_text_from_dataset(self, dataset_path):
        # Read the dataset from a CSV file
        try:
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"Dataset file not found: {dataset_path}")
            return

        # Check if 'text' column exists in the dataset
        if 'text' not in df.columns:
            print("Dataset must have a 'text' column.")
            return

        for idx, row in df.iterrows():
            inp_sent = row['text']
            print(f"Processing sentence {idx + 1}: {inp_sent}")
            
            # Process the text as per the existing methods
            lowercased_text = self.lower_case(inp_sent)
            print(f"Lowercased Text: {lowercased_text}")
            
            print(f"Tokens: {self.tokenize(lowercased_text)}")
            print(f"POS Tags: {self.POS(lowercased_text)}")
            
            isl_sentence = self.convert_to_isl(lowercased_text)
            print(f"ISL Sentence: {isl_sentence}\n")
    
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
        verb = ""
        objects = []
        adjectives = []
        question_word = ""
        possessive = ""
        adverbs = []
        punctuation = ""

        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text

            if token.pos_ == "VERB" and not verb:
                verb = token.text

            if token.dep_ in ("dobj", "pobj", "attr", "obj"):
                objects.append(token.text)

            if token.pos_ == "ADJ":
                adjectives.append(token.text)

            if token.tag_ in ("WP", "WRB"):
                question_word = token.text

            if token.dep_ == "poss":
                possessive = token.text

            if token.pos_ == "ADV":
                adverbs.append(token.text)

            if token.pos_ == "PUNCT":
                punctuation = token.text

        isl_sentence = []

        if possessive:
            isl_sentence.append(possessive)

        if subject:
            isl_sentence.append(subject)

        if adjectives:
            isl_sentence.extend(adjectives)

        if objects:
            isl_sentence.extend(objects)

        if adverbs:
            isl_sentence.extend(adverbs)

        if verb:
            isl_sentence.append(verb)

        if punctuation:
            isl_sentence.append(punctuation)

        return " ".join(isl_sentence)

import pickle

# Save the model to a .pkl file
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    text_to_ISL = TextToISL()
    # Save the model as 'text_to_isl_model.pkl'
    save_model(text_to_ISL, 'text_to_isl_model.pkl')
    # Now you can process the dataset or perform other operations
    text_to_ISL.process_text_from_dataset("your_dataset.csv")


# Load the model from the .pkl file
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Example usage
if __name__ == "__main__":
    # Load the saved model
    text_to_ISL = load_model('text_to_isl_model.pkl')
    # Use the loaded model to process the dataset
    text_to_ISL.process_text_from_dataset("your_dataset.csv")
