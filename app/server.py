from flask import Flask, request, jsonify
import joblib
import torch
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertModel


app = Flask(__name__)

print("Loading models...")
BEST_LOG_REG_TFIDF = joblib.load('models/simple/tfidf/cleaned_log_reg_model.joblib')
BEST_MNB_TFIDF = joblib.load('models/simple/tfidf/balanced_os_mnb_model.joblib')
BEST_DT_TFIDF = joblib.load('models/simple/tfidf/cleaned_dt_model.joblib')
BEST_RF_TFIDF = joblib.load('models/simple/tfidf/topic_merged_rf_model.joblib')
BEST_KNN_TFIDF = joblib.load('models/simple/tfidf/topic_merged_knn_model.joblib')

BEST_LOG_REG_DBERT = joblib.load('models/simple/distilbert_embed/topic_merged_lr_pca_bertembed_model.joblib')
BEST_MNB_DBERT = joblib.load('models/simple/distilbert_embed/cleaned_mnb_bertembed_model.joblib')
BEST_DT_DBERT = joblib.load('models/simple/distilbert_embed/cleaned_dt_bertembed_model.joblib')
BEST_RF_DBERT = joblib.load('models/simple/distilbert_embed/topic_merged_rf_bertembed_model.joblib')
BEST_KNN_DBERT = joblib.load('models/simple/distilbert_embed/topic_merged_knn_bertembed_model.joblib')

IDS_TO_LABELS = {0: 'Positive', 1: 'Negative', 2: 'Neutral', 3: 'Irrelevant'}
NUM_CLASSES = 4 
distilbert_classifier_layer = nn.Linear(768, NUM_CLASSES) 

STD_DISTILBERT_TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
CLEANED_DISTILBERT_MODEL = DistilBertModel.from_pretrained('models/distilbert_ft/train_cleaned')

SPECIAL_TOKENS = ['[TOPIC]', '[TWEET]']
CUSTOM_DISTILBERT_TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', additional_special_tokens=SPECIAL_TOKENS)

TOPIC_MERGED_DISTILBERT_MODEL = DistilBertModel.from_pretrained('models/distilbert_ft/train_topic_merged')
BALANCED_US_DISTILBERT_MODEL = DistilBertModel.from_pretrained('models/distilbert_ft/train_balanced_us')
BALANCED_OS_DISTILBERT_MODEL = DistilBertModel.from_pretrained('models/distilbert_ft/train_balanced_os')


models = {
    'log_reg_tfidf': BEST_LOG_REG_TFIDF,
    'mnb_tfidf': BEST_MNB_TFIDF,
    'dt_tfidf': BEST_DT_TFIDF,
    'rf_tfidf': BEST_RF_TFIDF,
    'knn_tfidf': BEST_KNN_TFIDF,

    'log_reg_distilbert': BEST_LOG_REG_DBERT,
    'mnb_distilbert': BEST_MNB_DBERT,
    'dt_distilbert': BEST_DT_DBERT,
    'rf_distilbert': BEST_RF_DBERT,
    'knn_distilbert': BEST_KNN_DBERT,

    'cleaned_distilbert': (STD_DISTILBERT_TOKENIZER, CLEANED_DISTILBERT_MODEL),
    'topic_merged_distilbert': (CUSTOM_DISTILBERT_TOKENIZER, TOPIC_MERGED_DISTILBERT_MODEL),
    'balanced_us_distilbert': (CUSTOM_DISTILBERT_TOKENIZER, BALANCED_US_DISTILBERT_MODEL),
    'balanced_os_distilbert': (CUSTOM_DISTILBERT_TOKENIZER, BALANCED_OS_DISTILBERT_MODEL)
}

distilbert_models = ['cleaned_distilbert', 'topic_merged_distilbert', 'balanced_us_distilbert', 'balanced_os_distilbert']
distilbert_embed_models = ['log_reg_distilbert', 'mnb_distilbert', 'dt_distilbert', 'rf_distilbert', 'knn_distilbert']


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    sentence = data['sentence']
    topic = data['topic']
    model = data['model']
    class_label = classify_sentence(sentence, topic, model)
    return jsonify({'class': class_label})


def classify_sentence(sentence, topic, model):
    if model in distilbert_models:
        tokenizer, selected_model = models[model]
        processed_sentence = preprocess(sentence, topic, model)

        with torch.no_grad():  # Inference mode
            outputs = selected_model(**processed_sentence)
            cls_representation = outputs.last_hidden_state[:, 0, :]
            logits = distilbert_classifier_layer(cls_representation)
            probabilities = torch.softmax(logits, dim=-1)
            prediction_idx = torch.argmax(probabilities, dim=-1).item()
        return IDS_TO_LABELS[prediction_idx]
    
    elif model in distilbert_embed_models:
        selected_model = models[model]
        processed_sentence = preprocess(sentence, topic, model)
        prediction = selected_model.predict(processed_sentence)
        return prediction[0]

    else:
        selected_model = models[model]
        processed_sentence = preprocess(sentence, topic, model)
        prediction = selected_model.predict([processed_sentence])
        return prediction[0]

def preprocess(sentence, topic, model):

    if model in ["mnb_tfidf", "rf_tfidf", "knn_tfidf"]: # these models require topic merging 
        sentence = topic + " " + sentence
    elif model in ["topic_merged_distilbert", "balanced_us_distilbert", "balanced_os_distilbert"]: # these models require tags & topic merging
        sentence = "[TOPIC] " + topic + " [TWEET] " + sentence
    
    # preprocessing for distilbert embed models
    if model in ["log_reg_distilbert", "mnb_distilbert", "dt_distilbert", "rf_distilbert", "knn_distilbert"]:
        embedder = TOPIC_MERGED_DISTILBERT_MODEL
        tokenizer = CUSTOM_DISTILBERT_TOKENIZER
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        outputs = embedder(**inputs)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        sentence = hidden_state.detach().cpu().numpy()

    # preprocessing for distilbert ft models
    if model in ["cleaned_distilbert", "topic_merged_distilbert", "balanced_us_distilbert", "balanced_os_distilbert"]:
        tokenizer, _ = models[model]
        sentence = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    return sentence

def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # cls token

def get_bert_representations(X):
    return torch.cat([get_bert_embeddings(text, CLEANED_DISTILBERT_MODEL, STD_DISTILBERT_TOKENIZER).detach().cpu() for text in X])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)