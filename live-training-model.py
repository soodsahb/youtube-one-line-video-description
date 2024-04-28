import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'facebook/bart-large-cnn'
bart_tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

nltk.download('punkt')
nltk.download('stopwords')


API_KEY = 'Youtube-api-key'


video_id = 'youtube-video-id'


comments = []
next_page_token = None

while True:
    
    url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100&textFormat=plainText'
    if next_page_token:
        url += f'&pageToken={next_page_token}'

    response = requests.get(url)
    data = response.json()

    if 'error' in data:
        raise Exception(f"API error: {data['error']['message']}")

  
    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

   
    next_page_token = data.get('nextPageToken')
    if not next_page_token:
        break

def preprocess_comment(comment):
  
    comment = re.sub(r'http\S+|www\S+', '', comment)
   
    comment = comment.encode('ascii', 'ignore').decode('ascii')
    
    comment = comment.translate(str.maketrans('', '', string.punctuation))
   
    comment = comment.lower()
   
    words = word_tokenize(comment)
   
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
   
    processed_comment = ' '.join(words)
    return processed_comment

processed_comments = []
for comment in comments:
    processed_comment = preprocess_comment(comment)
    processed_comments.append(processed_comment)


good_comments = [
"Great tutorial! It helped me understand the concepts very well.",
"The explanation was clear and concise. Thank you for the video!",
"I really enjoyed this video. The examples were very helpful.",
"This video made the topic much easier to understand. Appreciated!",
"Excellent content! I learned a lot from this video.",
"The step-by-step approach made it easy to follow along. Thanks!",
"I love how the instructor breaks down complex topics into simple terms.",
"This is exactly what I needed to know. Great job!",
"I appreciate the effort put into creating this valuable resource.",
"The video quality and audio are top-notch. Keep up the good work!",
"The visuals and animations really helped reinforce the concepts.",
"The instructor has a great teaching style that keeps me engaged.",
"This video is a lifesaver! It cleared up so many doubts.",
"The pacing of the video is perfect, not too fast or too slow.",
"I love the real-world examples used to illustrate the concepts.",
"The instructor's enthusiasm for the subject is contagious!",
"This video has given me the confidence to tackle this topic.",
"The level of detail in the explanations is just right.",
"The video is well-structured and easy to follow along.",
"The instructor has a great way of explaining complex ideas clearly.",
"Excellent tutorial! The concepts were explained in a clear and concise manner.",
"The video was incredibly helpful. The instructor's explanations were easy to understand.",
"I thoroughly enjoyed this video. The examples provided great clarity.",
"This video made a complex topic seem much more accessible. Well done!",
"Top-notch content! I gained a lot of valuable knowledge from this video.",
"The step-by-step approach made it easy to follow along. Great work!",
"I appreciate how the instructor simplified complex concepts without oversimplifying.",
"This video hit the nail on the head. It covered exactly what I needed to know.",
"A well-crafted resource that demonstrates the effort put into its creation.",
"The video and audio quality were outstanding, making for a great learning experience.",
"The visuals and animations really brought the concepts to life in an engaging way.",
"The instructor's teaching style kept me hooked from start to finish.",
"This video was a game-changer! It answered all my questions and more.",
"The pacing was perfect, allowing me to absorb the information at a comfortable rate.",
"The real-world examples used made it easier to relate to the concepts.",
"The instructor's enthusiasm for the subject matter was truly infectious.",
"Thanks to this video, I feel confident in my understanding of the topic.",
"The level of detail struck the perfect balance, avoiding being too broad or too granular.",
"The well-structured format of the video made it a breeze to follow along.",
"The instructor's engaging delivery style kept me hooked throughout the video.",
"The video provided a comprehensive overview of the subject matter.",
"The instructor's expertise shines through in this high-quality video.",
"I appreciate the clear and concise explanations provided in this video.",
"The video struck the perfect balance between theory and practical examples.",
"The instructor's passion for the topic made the video engaging and enjoyable.",
"This video is a must-watch for anyone looking to learn about this subject.",
"The video presents complex concepts in a way that is easy to understand.",
"I appreciated the logical flow and organization of the video's content.",
"The video's pace allowed me to digest the information without feeling rushed.",
"The instructor's clear communication skills made the video a pleasure to watch.",
"The video's production quality is top-notch, making it a joy to watch.",
"The video's focus on real-world applications made the content relatable.",
"The instructor's use of analogies and metaphors helped solidify my understanding.",
"This video is a valuable resource that I will refer back to time and again.",
"The video manages to strike the perfect balance between depth and accessibility.",
"The instructor's ability to break down complex concepts is truly impressive.",
"The video's format and structure made it easy to follow along and take notes.",
"The instructor's engaging presentation style kept me hooked throughout the video.",
"The video's use of visual aids and graphics enhanced the learning experience.",
"The instructor's clear and concise communication made the video a breeze to follow.",
"The video's focus on practicality and real-world relevance was greatly appreciated.",
"The instructor's enthusiasm for the subject was contagious and made the video enjoyable.",
"This video is an excellent resource for anyone looking to deepen their understanding.",
"The video's well-rounded approach covers all the essential aspects of the topic.",
"The instructor's clear explanations and examples made the content easy to grasp.",
"The video's pacing was just right, allowing for proper absorption of the material.",
"The instructor's engaging style and clear communication made the video a delight to watch.",
"The video's high production quality and attention to detail are commendable.",
"The instructor's use of real-world scenarios helped contextualize the concepts.",
"This video is a must-have resource for anyone serious about mastering the subject.",
"The video's comprehensive coverage left no stone unturned.",
"The instructor's ability to simplify complex topics is truly remarkable.",
"The video's structure and organization made it easy to follow and retain the information.",
"The instructor's engaging presentation style kept me hooked from start to finish.",
"The video's use of visuals and animations enhanced the learning experience.",
"The instructor's clear communication and concise explanations were greatly appreciated.",
"The video's focus on practical applications made the content immediately relevant.",
"The instructor's enthusiasm for the subject was contagious and made the video enjoyable.",
"This video is an excellent resource for anyone looking to expand their knowledge.",
"The video's comprehensive approach leaves no gaps in understanding.",
"The instructor's clear and concise explanations made the content easy to follow.",
"The video's pacing allowed for the proper absorption of the material.",
"The instructor's engaging style and clear communication made the video a joy to watch.",
"The video's high production quality and attention to detail are commendable.",
"The instructor's use of real-world examples helped solidify the concepts.",
"This video is an essential resource for anyone serious about mastering the subject."
]
bad_comments = [
"The audio quality was poor. Couldn't understand the explanation.",
"The video was too long and boring. Didn't learn much.",
"The examples were confusing. Needs better clarification.",
"Didn't find the video helpful. The content was too basic.",
"The instructor's voice is monotonous. Hard to stay engaged.",
"The pacing of the video is too slow. It could be more concise.",
"I expected more depth in the explanations. Disappointed.",
"The video didn't meet my expectations. Thumbs down.",
"The content is outdated. Not relevant anymore.",
"The video is poorly structured and hard to follow.",
"The audio quality made it challenging to focus on the content.",
"The video failed to hold my interest due to its excessive length.",
"The examples used were confusing and required further explanation.",
"The content was too basic and didn't provide enough depth.",
"The instructor's monotone delivery made it difficult to stay engaged.",
"The slow pacing of the video made it feel like it was dragging on.",
"The explanations lacked the depth I was hoping for, leaving me disappointed.",
"Unfortunately, this video did not live up to my expectations.",
"The outdated content made the video feel irrelevant and unhelpful.",
"The poor structure and organization made the video hard to follow.",
"The audio quality was subpar, making it difficult to understand the explanation.",
"The video dragged on unnecessarily, failing to hold my interest.",
"The examples used were confusing, requiring further clarification.",
"I didn't find the video particularly helpful. The content was too basic.",
"The instructor's monotonous voice made it challenging to stay engaged.",
"The pacing of the video was too slow, making it feel tedious to watch.",
"I was expecting more in-depth explanations, but was left disappointed.",
"This video didn't meet my expectations, and I wouldn't recommend it.",
"The content felt outdated and not relevant to current practices.",
"The lack of structure and organization made the video hard to follow.",
"The poor audio quality was a constant distraction throughout the video.",
"The excessive length of the video made it feel like a chore to watch.",
"The examples used were confusing and didn't help clarify the concepts.",
"The content was too basic and didn't offer anything new or insightful.",
"The instructor's monotone delivery made it difficult to stay focused.",
"The sluggish pacing of the video made it feel like it was dragging on forever.",
"The explanations lacked depth and left me with more questions than answers.",
"This video failed to meet my expectations and didn't deliver on its promises.",
"The outdated content made the video feel irrelevant and not worth my time.",
"The poor structure and organization made the video a frustrating experience.",
"The audio quality was so poor that it made the content hard to understand.",
"The video was unnecessarily long and failed to hold my attention.",
"The examples used were confusing and didn't help illustrate the concepts well.",
"The content was too basic and didn't offer anything new or insightful.",
"The instructor's monotonous voice made it challenging to stay engaged.",
"The pacing of the video was too slow, making it feel like a chore to watch.",
"The explanations lacked depth and left me feeling unsatisfied.",
"This video didn't meet my expectations and wasn't worth the time investment.",
"The outdated content made the video feel irrelevant and not worthwhile.",
"The poor structure and organization made the video a frustrating experience.",
"The audio quality was subpar, making it difficult to understand the content.",
"The excessive length of the video made it feel like a chore to watch.",
"The examples used were confusing and didn't help clarify the concepts.",
"The content was too basic and didn't offer any new insights.",
"The instructor's monotone delivery made it difficult to stay focused and engaged.",
"The sluggish pacing of the video made it feel like it was dragging on forever.",
"The explanations lacked depth and left me feeling unsatisfied.",
"This video failed to meet my expectations and wasn't worth my time.",
"The outdated content made the video feel irrelevant and not useful.",
"The poor structure and organization made the video a frustrating experience to follow."
]

processed_good_comments = [preprocess_comment(comment) for comment in good_comments]
processed_bad_comments = [preprocess_comment(comment) for comment in bad_comments]

sample_comments = processed_good_comments + processed_bad_comments
sample_labels = [1] * len(processed_good_comments) + [0] * len(processed_bad_comments)


train_comments, val_comments, train_labels, val_labels = train_test_split(
    sample_comments, sample_labels, test_size=0.2, random_state=42)


train_encodings = bert_tokenizer(train_comments, truncation=True, padding=True, max_length=128)
val_encodings = bert_tokenizer(val_comments, truncation=True, padding=True, max_length=128)


train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_labels)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_labels)
)


classifier_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
classifier_optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=2e-5)
classifier_model.train()


epochs = 10

for epoch in range(epochs):
    for batch in torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True):
        input_ids, attention_mask, labels = batch
        outputs = classifier_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        classifier_optimizer.step()
        classifier_model.zero_grad()


classifier_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(val_dataset, batch_size=8):
        input_ids, attention_mask, labels = batch
        outputs = classifier_model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Classifier Accuracy: {accuracy:.2f}")


good_comments = []
for comment in processed_comments:
    encoding = bert_tokenizer(comment, truncation=True, padding=True, max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = classifier_model(input_ids, attention_mask=attention_mask)
    prediction = torch.argmax(outputs.logits, dim=1)
    if prediction.item() == 1:
        good_comments.append(comment)


def prepare_input(comments):
    input_text = ' '.join(comments)
    input_text = input_text[:1024]  
    return input_text



def generate_description(input_text):
    inputs = bart_tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, max_length=50, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



good_comment_descriptions = []
for comment in good_comments:
    inputs = bart_tokenizer.encode(comment, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, max_length=50, early_stopping=True)
    description = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    good_comment_descriptions.append(description)

bad_comment_descriptions = []
for comment in bad_comments:
    inputs = bart_tokenizer.encode(comment, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, num_beams=4, max_length=50, early_stopping=True)
    description = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    bad_comment_descriptions.append(description)


sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')


input_text = prepare_input(good_comments)
one_line_description = generate_description(input_text)


one_line_embedding = sentence_model.encode([one_line_description])[0]
good_similarities = cosine_similarity([one_line_embedding], sentence_model.encode(good_comment_descriptions))[0]
bad_similarities = cosine_similarity([one_line_embedding], sentence_model.encode(bad_comment_descriptions))[0]

if sum(good_similarities) > sum(bad_similarities):
    final_description = max(zip(good_comment_descriptions, good_similarities), key=lambda x: x[1])[0]
    print("The video is good, and the matching description is:")
    print(final_description)
else:
    final_description = max(zip(bad_comment_descriptions, bad_similarities), key=lambda x: x[1])[0]
    print("The video is bad, and the matching description is:")
    print(final_description)