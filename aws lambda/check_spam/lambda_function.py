import json
import os
import requests
import boto3
import time
import telegram
from botocore.exceptions import ClientError
      
def lambda_handler(event, context):
    try:
        f = open('messages.json')
        reply_messages = json.load(f)
        request_body = json.loads(event['body'])
        f.close()
        
        chat_id = request_body['message']['chat']['id']
        message_id = request_body['message']['message_id']
        text = request_body['message']['text']
        text_list = list(text.split(" "))
    
        # parameters
        BOT_TOKEN = os.environ.get('TOKEN')
        BOT_CHAT_ID = chat_id 
        
        # for all messages starting with "/"
        if text_list[0][0] == "/":
            text_command = text_list[0][1:]
            if text_command == 'start':
                bot_message = reply_messages['start']
                send_text = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={BOT_CHAT_ID}&parse_mode=HTML&text={bot_message}'
                try:
                    response = requests.get(send_text)
                    response.raise_for_status()
                    print(f'Successfully sent message. Response: {response.text}')
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
            elif text_command == 'help':
                bot_message = reply_messages['help']
                send_text = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={BOT_CHAT_ID}&parse_mode=HTML&text={bot_message}'
                try:
                    response = requests.get(send_text)
                    response.raise_for_status()
                    print(f'Successfully sent message. Response: {response.text}')
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
            elif text_command == 'report':
                if 'reply_to_message' in request_body['message']:
                    reported_message_id = request_body['message']['reply_to_message']['message_id']
                    reported_message = request_body['message']['reply_to_message']['text']
                    bot_message = f"Reporting the following message: {reported_message}" 
                    send_text = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={BOT_CHAT_ID}&parse_mode=HTML&text={bot_message}'
                    try:
                        response = requests.get(send_text)
                        response.raise_for_status()
                        print(f'Successfully sent message. Response: {response.text}')
                    except requests.exceptions.RequestException as e:
                        print(f"An error occurred: {e}")
                    
                    # recording reported message
                    dynamodb = boto3.resource('dynamodb', region_name='ap-southeast-1')
                    table = dynamodb.Table('reported_messages')
                    increase_count(reported_message, table)
                    # update training data if report count == 10
                    if reached_threshold(reported_message, table):
                        headers = {"Content-Type": "application/json"}
                        data = {"text": reported_message}
                        try:
                            response = requests.post("https://deionna--spam-detector-add-spam-message-to-training-dataset.modal.run/", headers=headers, json=data)
                            response.raise_for_status()  # Raise an exception for non-200 status codes
                            # Handle successful response here
                            print(f"Successfully sent spam training data. Response: {response.text}")
                        except requests.exceptions.RequestException as e:
                            print(f"An error occurred: {e}")
                else:
                    bot_message = bot_message = reply_messages['report fail']
                    send_text = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={BOT_CHAT_ID}&parse_mode=HTML&text={bot_message}'
                    try:
                        response = requests.get(send_text)
                        response.raise_for_status()
                        print(f'Successfully sent message. Response: {response.text}')
                    except requests.exceptions.RequestException as e:
                        print(f"An error occurred: {e}")
        # for all messages starting without "/"
        else:
            headers = {"Content-Type": "application/json"}
            data = {"text": text}
            try:
                response = requests.post("https://deionna--spam-detector-predict.modal.run/", headers=headers, json=data)
                response.raise_for_status()  # Raise an exception for non-200 status codes
                # Handle successful response here
                print(f"Successfully sent message to Modal to check spam. Response: {response.text}")
                response_text = eval(response.text)
                print(f'Message detected as {response_text['predicted_class']}')
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
            if response_text['predicted_class'] == 'spam':
                try:
                    delete_message_url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage?chat_id={BOT_CHAT_ID}&message_id={message_id}"
                    response = requests.get(delete_message_url)
                    response.raise_for_status()
                     # Handle successful response here
                    print(f"Successfully deleted spam message. Response: {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"An error occurred: {e}")
        return {
            'statusCode': 200,
            'body': "Success"
        }
    except Exception as e:
        print(e)
        return {
            'statusCode': 200,
            'body': "Failed. Returned 200 to prevent jam in pending updates"
        }
    
    
    
def increase_count(message, table):
  try:
    response = table.get_item(Key={'message': message})
    if 'Item' in response:
      item = response['Item']
      count = item['count'] + 1
    else:
      count = 1  # New message, start count at 1
    table.put_item(Item={'message': message, 'count': count})
    return count == 10
  except ClientError as e:
    print(f"Error accessing DynamoDB: {e}")
    return False  # Indicate error

def reached_threshold(message, table):
  try:
    response = table.get_item(Key={'message': message})
    if 'Item' in response:
      item = response['Item']
      return item['count'] == 10
    else:
      return False  # Message not found, count is considered 0
  except ClientError as e:
    print(f"Error accessing DynamoDB: {e}")
    return False  # Indicate error