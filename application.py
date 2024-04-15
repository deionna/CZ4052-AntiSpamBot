from modal import Image, Stub, web_endpoint, Volume

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
stub = Stub("test")

vol = Volume.from_name("model")

@stub.function(image = image, volumes={"/model": vol})
@web_endpoint()
def predict(text):
    try:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
        import tensorflow as tf
        tokenizer = AutoTokenizer.from_pretrained("/model/model")
        model = TFAutoModelForSequenceClassification.from_pretrained("/model/model")
        # Tokenize the text
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
