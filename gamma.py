import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.initializers import Orthogonal
import pickle
from tensorflow.keras.layers import LSTM

# Custom LSTM that ignores 'time_major'
class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove 'time_major' if it exists
        super().__init__(*args, **kwargs)


# Load model and tokenizer
model = load_model('caption.h5', custom_objects={'Orthogonal': Orthogonal,'LSTM': CustomLSTM})  # Replace with your .h5 file path
tokenizer_path = 'tokenizer.pkl'  # Replace with your tokenizer path
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load VGG16 model for feature extraction
vgg_model = VGG16()
vgg_model = model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Parameters for generating captions
max_length = 34  # Set based on your trained model's max length
vocab_size = len(tokenizer.word_index) + 1

# Helper functions
def extract_features(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Streamlit app
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract features and generate caption
    feature = extract_features(image)
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    # Display the generated caption
    st.write("Generated Caption:", caption)
