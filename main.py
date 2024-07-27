# ### Import the Libraries
# import numpy as np 
# import tensorflow as tf 
# from tensorflow.keras.datasets import imdb 
# from tensorflow.keras.preprocessing import sequence 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Embedding , SimpleRNN ,Dense  
# from keras.models import load_model
# import warnings 
# warnings.filterwarnings('ignore')


# ### Load The IMDB dataset word Index 
# word_index=imdb.get_word_index()
# reverse_word_index={ value:key for key,value in word_index .items()}


# ### Load my per_Trained Model 
# try:
#     model = load_model('simple_rnn_imdb.h5')
#     model.summary()
#     model.get_weights()
# except AttributeError as e:
#     print(f"Error loading model: {e}")



# model.get_weights()

# ## Step 2 
# def decode_review(encoded_review):
#     return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])


# def preprocessing_text(text):
#     words=text.lower().split()
#     encoded_review=[word_index.get(word,2) + 3 for word in words]
#     padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
#     return padded_review



# ### Prediction Function 
# def predict_sentiment(review):
#    preprocessed_input=preprocessing_text(review)
#    prediction=model.predict(preprocessed_input)
#    sentiment='Posative' if prediction[0][0] >0.5 else 'Negative'
#    return sentiment , prediction[0][0] 


# #### Streamlit app
# import streamlit as st 

# st.title("IMDB Movie Review Sentiment Analysis")
# st.write('Enter a Movie  review to classify it as Posative or Negative')

# user_input=st.text_area('Movie Review')


# if st.button ('Classify'):
#    preprocess_input=preprocessing_text(user_input)
#    prediction=model.predict(preprocess_input)
#    sentiment='Posative' if prediction[0][0] >0.5 else 'Negative'


#    #Display the result 
#    st.write (f'Sentiment:{sentiment}')
#    st.write (f'Prediction Score:{prediction[0][0]}')

# else:
#     st.write(f'Please Enter A movie Review')







import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import warnings 
warnings.filterwarnings('ignore')

import streamlit as st 

# Load The IMDB dataset word Index 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load my pre-trained model 
try:
    model = load_model('simple_rnn_imdb.h5')
except AttributeError as e:
    st.error(f"Error loading model: {e}")

# Preprocessing function
def preprocessing_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function 
def predict_sentiment(review):
    preprocessed_input = preprocessing_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a Movie review to classify it as Positive or Negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment, score = predict_sentiment(user_input)
    # Display the result 
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please Enter A movie Review')
