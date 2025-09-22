import pandas as pd
import gradio as gr
import openai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone import ServerlessSpec
import google.generativeai as gemini
from google.generativeai import models
from google.generativeai import GenerativeModel, configure
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from groq import Groq  # Import Groq for Llama 3.3 and Deepseek models


# Define available LLM models
llm_models = {
    "GPT-3.5": "gpt-3.5-turbo-0125",
    "GPT-4": "gpt-4-turbo",
    "Gemini Pro": "gemini-1.5-flash",
    "Llama 3.3": "llama-3.3-70b-versatile",
    "Deepseek": "deepseek-r1-distill-llama-70b"
}

# Function to remove <think> blocks from responses
def remove_think_block(text: str) -> str:
    """
    Removes any content (including the tags) enclosed between <think> and </think>.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# Pinecone Index settings
INDEX_NAME = "match2st"
api_key = ''
pc = Pinecone(api_key=api_key)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(INDEX_NAME)

# Load CSV data
try:
    df = pd.read_csv("10Data_ST.csv", encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv("10Data_ST.csv", encoding='latin1')

# Calculate normalized weights for each selected criterion
def calculate_weights(description_weight, design_weight, inclusion_weight, exclusion_weight, selected_criteria):
    weight_map = {"low": 1, "medium": 2.5, "high": 4}
    selected_weights = { "description": weight_map[description_weight] if "description" in selected_criteria else 0,
                        "design": weight_map[design_weight] if "design" in selected_criteria else 0,
                        "inclusion": weight_map[inclusion_weight] if "inclusion" in selected_criteria else 0,
                        "exclusion": weight_map[exclusion_weight] if "exclusion" in selected_criteria else 0,
    }
    total_weight = sum(selected_weights.values())
    if total_weight > 0:
      normalized_weights = {k: (v / total_weight) * 100 for k, v in selected_weights.items()}
    else:
      normalized_weights = {k: 0 for k in selected_weights}
    return normalized_weights

def calculate_transformers_similarity(sentences1, sentences2):
    model_name='all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

   # Calculate the Euclidean distance between the two embeddings
    euclidean_distance = np.linalg.norm(np.array(embeddings1) - np.array(embeddings2))
    return euclidean_distance

def calculate_similarity_score_between_texts(text1, text2):
    openai_similarity = calculate_transformers_similarity([text1], [text2])
    return openai_similarity

# Calculate Average-Score for study
def calculate_Average_score(similarity_scores, selected_weights):
    score = 0
    score += similarity_scores["description"] * selected_weights["description"]
    score += similarity_scores["design"] * selected_weights["design"]
    score += similarity_scores["inclusion"] * selected_weights["inclusion"]
    score += similarity_scores["exclusion"] * selected_weights["exclusion"]

    return score


# RAG (Retrieve Augment Generate) Function
def RAG(prompt_text, top_k, weights, input_study, selected_weights):
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(prompt_text)
    context, matched_trials = gradio_interface(embeddings, top_k, input_study, selected_weights)
    return context if context else "No similar studies found.", matched_trials

# Gradio interface for matched studies
def gradio_interface(context, top_k, input_study, selected_weights):
    if df is None:
        return "No similar studies found.", ""

    vec = context.tolist()
    response = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True
    )
    all_matches = response['matches']

    ret = "\nHere is the first set of Study Title, Brief Summary, Inclusion Criteria, and Exclusion Criteria: "
    matched_trials = ""

    for j, match in enumerate(all_matches):
        i = int(match['id'])
        similarity_score = match['score']

        # Calculate similarity scores for each criterion
        description_similarity = calculate_similarity_score_between_texts(input_study["Study Description"], df['Brief Summary'][i])
        design_similarity = calculate_similarity_score_between_texts(input_study["Study Design"], df['Study Design'][i])
        inclusion_similarity = calculate_similarity_score_between_texts(input_study["Inclusion Criteria"], df['Inclusion Criteria'][i])
        exclusion_similarity = calculate_similarity_score_between_texts(input_study["Exclusion Criteria"], df['Exclusion Criteria'][i])

        similarity_scores = {
            "description": description_similarity,
            "design": design_similarity,
            "inclusion": inclusion_similarity,
            "exclusion": exclusion_similarity
        }

        Average_score = calculate_Average_score(similarity_scores, selected_weights)

        r = f"\nSTUDY NO.: {j+1}"
        r += f"\nNCT Number: {df['NCT Number'][i]}"
        r += f"\nSimilarity Score (Euclidean Distance): {similarity_score:.2f}"
        r += f"Study Description Similarity: {description_similarity:.2f}\n"
        r += f"Study Design Similarity: {design_similarity:.2f}\n"
        r += f"Inclusion Criteria Similarity: {inclusion_similarity:.2f}\n"
        r += f"Exclusion Criteria Similarity: {exclusion_similarity:.2f}\n"
        r += f"Average-Score: {Average_score:.2f}\n"
        r += f"\nStudy Title: {df['Study Title'][i]}"
        r += f"\nBrief Summary: {df['Brief Summary'][i]}"
        r += f"\nEnrollment: {df['Enrollment'][i]}"
        r += f"\nStart Date: {df['Start Date'][i]}"
        r += f"\nCompletion Date: {df['Completion Date'][i]}"
        r += f"\nInclusion Criteria: {df['Inclusion Criteria'][i]}"
        r += f"\nExclusion Criteria: {df['Exclusion Criteria'][i]}\n"

        matched_trials += r + "\n"

        if j != top_k-1:
            r += " \nHere is another set of Study Title, Brief Summary, Enrollment, Start Date, Completion Date, Inclusion Criteria, and Exclusion Criteria: "

        ret += r

    return ret if ret else "No similar studies found.", matched_trials

# Declare global variables
similar_trials = ""

# Process LLM requests for different models
def process_gemini(query, api_key):
    configure(api_key=api_key)
    model = GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(query)
        response_text = response.text if hasattr(response, 'text') else str(response)
        return response_text
    except Exception as e:
        return f"Error with Gemini API: {str(e)}"

def process_llama(query, api_key, temp_in):
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=llm_models["Llama 3.3"],
            messages=[
                {"role": "system", "content": "You are an expert in analyzing clinical trials."},
                {"role": "user", "content": query}
            ],
            temperature=temp_in,
            max_tokens=4000,
            top_p=1,
            stream=False,
            stop=None
        )
        response_text = completion.choices[0].message.content
        return response_text
    except Exception as e:
        return f"Error with Llama API: {str(e)}"

def process_deepseek(query, api_key, temp_in):
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=llm_models["Deepseek"],
            messages=[
                {"role": "system", "content": "You are an expert in analyzing clinical trials."},
                {"role": "user", "content": query}
            ],
            temperature=temp_in,
            max_tokens=4000,
            top_p=1,
            stream=False,
            stop=None
        )
        response_text = completion.choices[0].message.content
        # Remove any chain-of-thought blocks enclosed in <think>...</think>
        response_text = remove_think_block(response_text)
        return response_text
    except Exception as e:
        return f"Error with Deepseek API: {str(e)}"

def process_gpt(query, api_key, model, temp_in):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing clinical trials."},
                {"role": "user", "content": query}
            ],
            temperature=temp_in,
            max_tokens=4000
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error with GPT API: {str(e)}"

def process_text(input1, input2, input3, input4, exclude_input1, exclude_input2, exclude_input3, exclude_input4, api_key, model_in, temp_in, rag_check, top_k,
                 description_weight, design_weight, inclusion_weight, exclusion_weight, selected_criteria):

    global similar_trials  # Declare this as a global variable

    weight_map = {"low": 1, "medium": 2.5, "high": 4}
    selected_weights = { "description": weight_map[description_weight] if "description" in selected_criteria else 0,
                        "design": weight_map[design_weight] if "design" in selected_criteria else 0,
                        "inclusion": weight_map[inclusion_weight] if "inclusion" in selected_criteria else 0,
                        "exclusion": weight_map[exclusion_weight] if "exclusion" in selected_criteria else 0,
    }

    # Normalize weights based on the selected criteria and their weights
    weights = calculate_weights(description_weight, design_weight, inclusion_weight, exclusion_weight, selected_criteria)

    # input study
    input_study = {
        "Study Description": input1,
        "Inclusion Criteria": input2,
        "Exclusion Criteria": input3,
        "Study Design": input4
    }

    # Construct the string that will be passed to the LLM
    curr_string = "Based on the past trials, please provide similar studies considering the following exclude criteria:\n"

    if exclude_input1:
        curr_string += f"Exclude Study Description:\n{exclude_input1}\n"
    if exclude_input2:
        curr_string += f"Exclude Inclusion Criteria:\n{exclude_input2}\n"
    if exclude_input3:
        curr_string += f"Exclude Exclusion Criteria:\n{exclude_input3}\n"
    if exclude_input4:
        curr_string += f"Exclude Study Design:\n{exclude_input4}\n"

    similar_trials = ""  # Reset for each new call
    if rag_check:
        rag_string = f"Study Description: {input1}\nInclusion Criteria: {input2}\nExclusion Criteria: {input3}\nStudy Design: {input4}"
        query, matched_trials = RAG(rag_string, top_k, weights, input_study, selected_weights)
        similar_trials = matched_trials
        curr_string += "\nTo help you generate improved criteria, consider the following similar studies:\n" + query

    # Generate verdict based on the selected model
    verdict = generate_llm_verdict(api_key, model_in, curr_string, weights, temp_in, similar_trials)
    return verdict, similar_trials

def generate_llm_verdict(api_key, model_in, query, weights, temperature, similar_trials):
    # Format the weight information for the prompt
    weight_string = ", ".join([f"{criterion.capitalize()} Weight: {weight:.2f}%" for criterion, weight in weights.items()])
    
    # Construct the full prompt with instructions
    full_prompt = f""" You are an expert in analyzing clinical trials. Certain similar trials have been provided to you (Set A).
                    Also, you have been provided with exclude descriptions against different study components, like study description ("Exclude Study Description"), study design ("Exclude Study Design")
                    , inclusion criteria ("Exclude Inclusion Criteria") , and exclusion criteria ("Exclude Exclusion Criteria").
                    
                    Your task is as follows:
                    
                    Strictly apply the exclude criteria:
                    If any of the "Exclude Study Description" , "Exclude Inclusion Criteria" , "Exclude Exclusion Criteria" , or "Exclude Study Design" becomes true for a study in set A, then you need
                    to remove the specific study from set A.
                      
                    If any exclude criteria involve ranges or numerical thresholds (e.g., BMI ≥ 27 and ≤ 55 kg/m², BMI >= 25 kg/m²), studies that meet the exclude criteria must be excluded.
                    So, the output will contain those studies in set A where the exclude criteria is false. Also, the output needs to contain the reason against each study excluded in set A where
                    the exclude criteria were true.
                    For example:
                    
                    If the "Exclude Study Description" , "Exclude Inclusion Criteria" , "Exclude Exclusion Criteria" , or "Exclude Study Design" , specifies
                    BMI will exceed 20 kg/m², any study with an inclusion BMI range entirely
                    above 20 kg/m² (e.g., BMI ≥ 25 kg/m²) or a range partially exceeding 20 kg/m² (e.g., BMI ≥ 18 kg/m² to 30 kg/m²) should be excluded from Set A.
                    If the "Exclude Study Description" , "Exclude Inclusion Criteria" , "Exclude Exclusion Criteria" , or "Exclude Study Design" specifies a specific range,
                    such as BMI ≥ 27 and ≤ 55 kg/m², studies whose inclusion BMI ranges overlap or match this range
                    must be excluded from Set A.
                    So if the Exclude is partially true then exclude the study from Set A.
                    
                    For exclusion descriptions like "Age 18 years old or older," studies with inclusion criteria starting at age 18 or higher should be excluded.
                    Now, in the remaining studies in set A after applying the exclude criteria we need to do the following:
                    Classify the remaining studies:
                    Using the average similarity score provided for each study in Set A, classify them as:
                    
                    Low: Average-Score <= 5
                    Medium: 5 < Average-Score <= 10
                    High: Average-Score > 10
            
            
                    Output the reasons for exclusion:
                    For each excluded study, provide a clear and concise explanation of why it was excluded, including the matching criteria and description.
                    
                    Input Format:
                    
                    Excludes descriptions:
                    
                    Against study description: {{exclude_input1}}
                    Against inclusion criteria: {{exclude_input2}}
                    Against exclusion criteria: {{exclude_input3}}
                    Against study design: {{exclude_input4}}
                    
                    Set A:
                    {query}
                    
                    The output from you needs to be formatted this way:
                    
                    Similar studies before excludes:
                    <all studies passed to you in Set A with their NCT IDs>
                    
                    Similar studies after excludes with their similarity rating "Low," "Medium," or "High":
                    <studies in Set A with their NCT IDs after applying excludes>
                    
                    Similar studies excluded with reasoning:
                    <studies excluded from Set A with their NCT IDs as a result of excludes description being true and the reason in brief as to why they were excluded – give a reason for each study excluded>
                    """

    # Route to the correct model processor
    if model_in == "Gemini Pro":
        return process_gemini(full_prompt, api_key)
    elif model_in == "Llama 3.3":
        return process_llama(full_prompt, api_key, temperature)
    elif model_in == "Deepseek":
        return process_deepseek(full_prompt, api_key, temperature)
    elif model_in == "GPT-3.5":
        return process_gpt(full_prompt, api_key, llm_models["GPT-3.5"], temperature)
    else:  # Default to GPT-4
        return process_gpt(full_prompt, api_key, llm_models["GPT-4"], temperature)


# Clear inputs
def clear_inputs():
    return "", "", "", "", "", "", "", "", "", "GPT-4", 0.9, True, 2, "medium", "medium", "medium", "medium", ["description", "design", "inclusion", "exclusion"]


# Gradio interface
with gr.Blocks() as demo:
    gr.HTML("""<div style="text-align: center;">
               <img src="file/assets/logo.png" alt="logo" width="150" height="150">
                <h1>Similarity Tool</h1>
                </div>""")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input1 = gr.Textbox(label="Study Description", lines=1, placeholder="Enter Study Description")
                with gr.Column():
                    exclude_input1 = gr.Textbox(label="Exclude Study Description (if any)", lines=1, placeholder="Enter text to exclude")
            with gr.Row():
                with gr.Column():
                    input2 = gr.Textbox(label="Inclusion Criteria", lines=1, placeholder="Enter Inclusion Criteria")
                with gr.Column():
                    exclude_input2 = gr.Textbox(label="Exclude Inclusion Criteria (if any)", lines=1, placeholder="Enter text to exclude")
            with gr.Row():
                with gr.Column():
                    input3 = gr.Textbox(label="Exclusion Criteria", lines=1, placeholder="Enter Exclusion Criteria")
                with gr.Column():
                    exclude_input3 = gr.Textbox(label="Exclude Exclusion Criteria (if any)", lines=1, placeholder="Enter text to exclude")
            with gr.Row():
                with gr.Column():
                    input4 = gr.Textbox(label="Study Design", lines=1, placeholder="Enter Study Design")
                with gr.Column():
                    exclude_input4 = gr.Textbox(label="Exclude Study Design (if any)", lines=1, placeholder="Enter text to exclude")

            api_key = gr.Textbox(label="API Key", placeholder="Enter API Key")
            model_in = gr.Dropdown(choices=["GPT-4", "GPT-3.5", "Gemini Pro", "Llama 3.3", "Deepseek"], value="GPT-4", label="LLM Model")
            temp_in = gr.Slider(label="Temperature", value=0.9, minimum=0, maximum=1)

            rag_check = gr.Checkbox(value=True, label="Apply RAG")
            top_k = gr.Slider(label="Number of Similar Trials to Retrieve (k)", value=2, minimum=1, maximum=10)

            description_weight = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Description Weight")
            design_weight = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Design Weight")
            inclusion_weight = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Inclusion Weight")
            exclusion_weight = gr.Dropdown(choices=["low", "medium", "high"], value="medium", label="Exclusion Weight")

            selected_criteria = gr.CheckboxGroup(label="Select Criteria for Weighting", choices=["description", "design", "inclusion", "exclusion"], value=["description", "design", "inclusion", "exclusion"])

            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            output1 = gr.Textbox(label="LLM Verdict", lines=20)
            output2 = gr.Textbox(label="Similar Trials", lines=20)

    submit_btn.click(process_text,
                     inputs=[input1, input2, input3, input4, exclude_input1, exclude_input2, exclude_input3, exclude_input4, api_key, model_in, temp_in, rag_check, top_k,
                             description_weight, design_weight, inclusion_weight, exclusion_weight, selected_criteria],
                     outputs=[output1, output2])

    clear_btn.click(clear_inputs, outputs=[input1, input2, input3, input4, exclude_input1, exclude_input2, exclude_input3, exclude_input4, api_key, model_in, temp_in, rag_check, 
                                           top_k, description_weight, design_weight, inclusion_weight, exclusion_weight, selected_criteria])

demo.launch(debug=True)
