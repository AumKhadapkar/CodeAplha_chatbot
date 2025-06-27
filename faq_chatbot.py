import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

faq_data = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any product within 30 days of purchase."
    },
    {
        "question": "How can I track my order?",
        "answer": "Use the tracking link sent to your email after shipping."
    },
    {
        "question": "Do you offer customer support?",
        "answer": "Yes, we offer 24/7 customer support via chat and email."
    },
    {
        "question": "What payment methods are accepted?",
        "answer": "We accept credit cards, debit cards, UPI, and net banking."
    }
]

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

questions = [preprocess(faq['question']) for faq in faq_data]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score > 0.3:
        return faq_data[best_match]["answer"]
    else:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing your question."

print("🤖 FAQ Chatbot - Ask me anything! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
