import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key='')

filename = '/Users/bill/PycharmProjects/K means/OpenAI_Parsing/vqa_processed.json'
with open(filename, 'r') as file:
    data = json.load(file)

data = data[0:6]

def generate_new_questions(item):

    generated_questions = []
    question =item.get('question')
    answers = item.get('answers', [])

    # Constructing the prompt for the API
    base_prompt= "Please create one more suitable question based on the given Q&A pair. " \
                  "Only provide the question:"

    for answer_group in answers:
        answer_text = " / ".join([a for sublist in answer_group for a in sublist])
        prompt= f"{base_prompt} Question: {question} Answer: {answer_text}"
        # Sending prompt to the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are an assistant trained to improve question phrasing."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_questions.append(response.choices[0].message.content)

    return generated_questions

for item in tqdm(data):
    new_questions = generate_new_questions(item)
    item['generated_questions'] = new_questions

output_filename = 'new_questions.json'
with open(output_filename, 'w') as outfile:
    json.dump(data, outfile, indent=4)

print(f"All data has been processed and saved to {output_filename}.")
