import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
import time
import json
from groq import Groq
from openai import OpenAI
import anthropic
import base64
import pandas as pd

def encode_image(image_path):
    with open(__location__ + "/" + image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_groq_model(base64_image, prompt):
    client = Groq()
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llava-v1.5-7b-4096-preview",
    )
    end_time = time.time()
    return chat_completion.choices[0].message.content, end_time - start_time

def run_openai_model(base64_image, prompt):
    client = OpenAI()
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    end_time = time.time()
    return response.choices[0].message.content, end_time - start_time

def judge_responses(ground_truth, groq_response, openai_response, groq_prompt, openai_prompt):
    client = anthropic.Anthropic()
    judge_prompt = f"""
    You are an impartial judge evaluating the performance of two AI vision models. 
    You will be provided with:
    1. The ground truth about an image
    2. The responses from two different models
    3. The prompts given to both models

    Your task is to evaluate and score each model's response based on the following criteria:
    - Accuracy (0-10): How well does the response align with the ground truth?
    - Completeness (0-10): How thorough is the response in addressing all aspects of the prompt?
    - Relevance (0-10): How relevant is the response to the given prompt?
    - Insight (0-10): Does the response provide any unique or insightful observations?

    For each of Groq and OpenAI responses, please provide a score between 0 and 10 for each of the criteria above. Then also add a total score which is the sum of all the scores and a short comment on the evaluation.

    Here is the information for your evaluation:
    Ground Truth:
    <context>
    {ground_truth}
    </context>
    
    Groq Prompt: 
    <prompt>
    {groq_prompt}
    </prompt>
    
    Groq Response: 
    <response>
    {groq_response}
    </response>
    
    OpenAI Prompt:
    <prompt>
    {openai_prompt}
    </prompt>
    
    OpenAI Response: 
    <response>
    {openai_response}
    </response>
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        # temperature=0.,
        messages=[
            {"role": "user", "content": judge_prompt}
        ],
        tools = [
            {
                "name": "evaluation_of_vision_model_responses",
                "description": "Evaluate the responses of two vision models based on accuracy, completeness, relevance, and insight.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "groq_evaluation": {
                            "type": "object",
                            "properties": {
                                "accuracy": {"type": "number"},
                                "completeness": {"type": "number"},
                                "relevance": {"type": "number"},
                                "insight": {"type": "number"},
                                "total": {"type": "number"},
                                "comments": {"type": "string"}
                            }
                        },
                        "openai_evaluation": {
                            "type": "object",
                            "properties": {
                                "accuracy": {"type": "number"},
                                "completeness": {"type": "number"},
                                "relevance": {"type": "number"},
                                "insight": {"type": "number"},
                                "total": {"type": "number"},
                                "comments": {"type": "string"}
                            }
                        }
                    },
                    "required": ["groq_evaluation", "openai_evaluation"]
                },
            }
        ],
        tool_choice = {"type": "tool", "name": "evaluation_of_vision_model_responses"}
    )
    
    return response.content[0].input


def generate_report(results):
    report = "Vision Model Comparison Report\n"
    report += "==============================\n\n"
    
    for image, image_results in results.items():
        report += f"Image: {image}\n"
        report += "--------------------\n"
        
        for result in image_results:
            report += f"Groq Prompt: {result['groq_prompt']}\n"
            report += f"OpenAI Prompt: {result['openai_prompt']}\n\n"
            
            report += "Groq Model:\n"
            report += f"Response: {result['groq']['response']}\n"
            report += f"Time: {result['groq']['time']:.2f}s\n"
            report += "Evaluation:\n"
            for key, value in result['groq']['evaluation'].items():
                report += f"  {key.capitalize()}: {value}\n"
            report += "\n"
            
            report += "OpenAI Model:\n"
            report += f"Response: {result['openai']['response']}\n"
            report += f"Time: {result['openai']['time']:.2f}s\n"
            report += "Evaluation:\n"
            for key, value in result['openai']['evaluation'].items():
                report += f"  {key.capitalize()}: {value}\n"
            report += "\n"
            
            report += "-" * 50 + "\n\n"
        
        report += "=" * 50 + "\n\n"
    
    return report
