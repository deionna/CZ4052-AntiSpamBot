import requests

def lambda_handler(event, context):
    url = "https://deionna--spam-detector-retrain.modal.run/"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        # Handle successful response here
        return {
            'statusCode': 200,
            'body': response.text  # Example: Return the response content
        }

    except requests.exceptions.RequestException as e:
        return {
            'statusCode': 500,
            'body': f"An error occurred: {e}"
        }
