from modal import Image, Stub, web_endpoint, Volume
from typing import Dict

image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "tensorflow==2.16.1",
        "transformers==4.39.3",
        "tf_keras==2.16.0"
    )
)

stub = Stub("stable-diffusion-xl")

# Load tokenizer and model
stub = Stub("spam_detector")

vol = Volume.from_name("model")

@stub.function(image = image, volumes={"/model": vol})
@web_endpoint(method="POST")
def predict(body: Dict):
    try:
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
    except Exception as e:
        return str(e)

# Retraining function
@stub.function(image=image, volumes={"/model": vol})
def retrain():
    try:
        import pandas as pd
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        import tensorflow as tf

        # Load dataset using pandas
        df = pd.read_csv('/model/model/training_data.csv')
        texts = df['sms'].tolist()
        labels = df['label'].tolist()

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("/model/model")
        model = TFAutoModelForSequenceClassification.from_pretrained("/model/model")

        # Tokenize the dataset
        train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            labels
        ))

        # Prepare for training: define loss, optimizer, metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        # Training
        model_history = model.fit(train_dataset.shuffle(1000).batch(8), epochs=3)

        # Save model
        model.save_pretrained("/model/model")  # Saving it back to the mounted volume

        return "Model retrained successfully"
    except Exception as e:
        return str(e)
