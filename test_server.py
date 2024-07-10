# create a series of json requests to test the server

import requests

def test_server():
    url = "http://127.0.0.1:5000/classify"
    headers = {'Content-Type': 'application/json'}
    
    models = ['log_reg_tfidf', 'mnb_tfidf', 'dt_tfidf', 'rf_tfidf', 'knn_tfidf', 'log_reg_distilbert', 'mnb_distilbert', 'dt_distilbert', 'rf_distilbert', 'knn_distilbert', 'cleaned_distilbert', 'topic_merged_distilbert', 'balanced_us_distilbert', 'balanced_os_distilbert']

    sentence = "I don't like FIFA24."
    topic = "FIFA"

    for model in models:
        data = {
            'sentence': sentence,
            'topic': topic,
            'model': model
        }
        response = requests.post(url, headers=headers, json=data)
        print(f"Model: {model}, Prediction: {response.json()['class']}")

if __name__ == "__main__":
    test_server()