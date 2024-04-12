from openai import OpenAI
import json
import pdb
from tqdm import tqdm

client = OpenAI(api_key='')

filename = '/OpenAI_Parsing/VQA_ans_diff_train.json'
with open(filename, 'r') as file:
    data = json.load(file)

# Filter the data
# Only 22 json satisfies the condition
filtered_data = [x for x in data if x['ans_diff_labels'][-3] >= 5]
print(len(filtered_data))
# data = filtered_data[0:10]

example = """
 {
  “qid”: “441361005",
  “image”: “COCO_train2014_000000441361.jpg”,
  “question”: “Who is on the bench?“,
  “src_dataset”: “VQA”,
  “answers”: [
    [“old women”, “women”],
    [“woman and man”, “couple”, “elderly couple”],
    [“2 people”, “people”],
    [“at park”]
  ],
  “ans_diff_labels”: [
    0,
    0,
    0,
    0,
    5,
    0,
    3,
    5,
    0,
    0
  ]
}
"""

# Function to format the message content for the API call
def format_message_content(item):
    json_string = json.dumps(item, indent=4)
    message_content = ("I will provide part of a JSON format information, please "
                       "convert unique answers into “clusters” of answers, where "
                       "each “cluster” (set of answers that mean the same thing) "
                       "are in a different line. Please only keep the distinct "
                       "answer. Please don't delete the 'ans_diff_labels'. "
                       "Here is the data: " + json_string + "Here is an example" + example)
    return message_content


# List to collect all responses
all_responses = []

for item in tqdm(filtered_data):
    # Format the message content for each item
    message_content = format_message_content(item)

    # Make the API call for each item
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": message_content}
        ]
    )

    # Extract the response content and add it to the all_responses list
    response_content = response.choices[0].message.content
    # pdb.set_trace()
    all_responses.append(json.loads(response_content))

    # Write all_responses to a single JSON file
    output_filename = 'vqa_processed.json'
    with open(output_filename, 'w') as outfile:
        json.dump(all_responses, outfile, indent=4)

print(f"All responses have been processed and saved to {output_filename}.")
