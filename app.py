from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import gradio as gr

 
tokenizer_sentence_analysis = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model_sentence_analysis = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

tokenizer_review_feedback_sentiment = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model_review_feedback_sentiment = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def sentence_sentiment_model(text, tokenizer, model):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        result = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = result.logits.detach()
        probs = torch.softmax(logits, dim=1)
    pos_prob = probs[0][2].item()
    neu_prob = probs[0][1].item()  
    neg_prob = probs[0][0].item() 
    return {'Positive': [round(float(pos_prob), 2)],"Neutural":[round(float(neu_prob), 2)], 'Negative': [round(float(neg_prob), 2)]}

def review_feedback_sentiment(text, tokenizer, model):
    inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors="pt")
    with torch.no_grad():
        result = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = result.logits.detach()
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        categories = ['Terrible', 'Poor', 'Average', 'Good', 'Excellent']
        output_dict = {}
        for i in range(len(categories)):
            output_dict[categories[i]] = [round(float(probs[i]), 2)]
    return output_dict



def emotion_sentiment(text):
    results = classifier(text, padding='max_length', max_length=512)
    return {label['label']: [label['score']] for label in results[0]}
 


def sentence_analysis(text):
    result = sentence_sentiment_model(text,tokenizer_sentence_analysis,model_sentence_analysis)
    return result
def emotion(text):
    result = emotion_sentiment(text)
    return result
def review_feed_back(text):
    result = review_feedback_sentiment(text,tokenizer_review_feedback_sentiment,model_review_feedback_sentiment)
    return result

def selection_model(model,text):
    if text == "":
        return "No Text Input"
    if model=="Emotion Analysis":
        return emotion(text)
    if model == "Review Feedback Analysis":
        return review_feed_back(text)
    if model == "Sentence Analysis":
        return sentence_analysis(text)
    return "Please select model"


paragraph = """
I woke up this morning feeling refreshed and excited for the day ahead. 
""" 

with gr.Blocks(title="Sentiment",css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Emotion, ReviewFeedback, Sentence Analysis")
            with gr.Row():           
                with gr.Column():
                    drop_down_name = gr.Dropdown(choices=["Emotion Analysis", "Review Feedback Analysis", "Sentence Analysis"],label="Model")
                    inputs = gr.TextArea(label="sentence",value=paragraph,interactive=True)
                    btn = gr.Button(value="RUN")
                with gr.Column():
                    output = gr.Label(label="output") 
                btn.click(fn=selection_model,inputs=[drop_down_name,inputs],outputs=[output])
demo.launch() 

 