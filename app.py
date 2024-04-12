from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import boto3
from botocore.exceptions import NoCredentialsError

app = Flask(__name__)

# Download model from S3
def download_model_from_s3(bucket_name, model_key, download_path):
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, model_key, download_path)
        print("Model downloaded successfully")
    except NoCredentialsError:
        print("Credentials not available")
        raise
    except Exception as e:
        print("Error downloading model from S3:", e)
        raise

# Load tokenizer and model
def load_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    return tokenizer, model

# Replace this with your S3 location
MODEL_S3_BUCKET = "spammodel"
MODEL_S3_KEY = "model"

download_path = 'model'  # Temporary path for downloading the model files
download_model_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY, download_path)
tokenizer, model = load_model(download_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract text input from request
        text = request.json['text']

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
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
