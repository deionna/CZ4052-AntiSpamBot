from modal import Image, Stub, web_endpoint, Volume
from typing import Dict

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "tensorflow==2.16.1",
        "transformers==4.39.3",
        "tf_keras==2.16.0",
        "pandas==1.5.3"
    )
)

stub = Stub("stable-diffusion-xl")

# Load tokenizer and model
stub = Stub("spam_detector")

vol = Volume.from_name("model")

@stub.function(image = image, volumes={"/model": vol})
@web_endpoint(method="POST")
def predict(body: Dict):
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    import tensorflow as tf
    tokenizer = AutoTokenizer.from_pretrained("/model/model")
    model = TFAutoModelForSequenceClassification.from_pretrained("/model/model")
    # Tokenize the text
    text = body['text']
    inputs = tokenizer(text, return_tensors="tf", truncation=True, max_length=512)

    # Predict
    outputs = model(inputs)
    logits = outputs.logits

    # Use softmax to get the probabilities
    probs = tf.nn.softmax(logits, axis=-1)

    # Get the predicted class
    predicted_class = tf.argmax(probs, axis=-1).numpy()[0]

    # Return prediction result
    result = {
        'predicted_class': 'spam' if predicted_class == 1 else 'ham',
        'probability': float(probs[0][predicted_class].numpy())
    }
    return result

@stub.function(image = image, volumes={"/model": vol})
@web_endpoint(method="POST")
def add_spam_message_to_training_dataset(body: Dict):
    import pandas as pd
    text = body['text']

    df = pd.read_csv("/model/model/training_data.csv")
    new_data = {'sms': text, 'label': 1}
    df = df.append(new_data, ignore_index=True)
    df.to_csv("/model/model/training_data.csv")
    
    return "Success"

    


