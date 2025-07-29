import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer (can be changed to any fine-tuned character model)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sentiment analyzer
sentiment_pipeline = pipeline("sentiment-analysis")

# Prompt templates
emotion_templates = {
    "POSITIVE": "You're a cheerful and encouraging NPC. Reply brightly to this message: ",
    "NEGATIVE": "You're a sad but wise NPC. Respond with gentle support to: ",
    "NEUTRAL": "You're a calm and neutral NPC. Give a straightforward response to: "
}

# Interactive loop
def generate_dialog(user_input):
    sentiment = sentiment_pipeline(user_input)[0]['label']
    base_prompt = emotion_templates.get(sentiment, emotion_templates["NEUTRAL"])
    prompt = base_prompt + user_input

    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    print("Sentient Dialog is running. Type your message to an NPC. Type 'exit' to quit.")
    while True:
        user = input("You: ")
        if user.strip().lower() == "exit":
            break
        reply = generate_dialog(user)
        print("NPC:", reply)
