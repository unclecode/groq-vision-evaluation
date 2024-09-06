import os, sys

# append parent directory to import from comparision_common.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
import time
import json
from groq import Groq
from openai import OpenAI
import anthropic
import base64
import pandas as pd
from utils import *
from analysis import *


def compare_models(image_path, prompt, ground_truth):
    base64_image = encode_image(image_path)

    print(f"[LOG] Call to Groq model {image_path}.")
    groq_response, groq_time = run_groq_model(base64_image, prompt)

    print(f"[LOG] Call to OpenAI model {image_path}.")
    openai_response, openai_time = run_openai_model(base64_image, prompt)

    print(f"[LOG] Evaluating responses for image: {image_path}")
    evaluation = judge_responses(
        ground_truth, groq_response, openai_response, prompt, prompt
    )

    return {
        "prompt": prompt,
        "groq": {
            "response": groq_response,
            "time": groq_time,
            "evaluation": evaluation["groq_evaluation"],
        },
        "openai": {
            "response": openai_response,
            "time": openai_time,
            "evaluation": evaluation["openai_evaluation"],
        },
    }


def save_results_to_csv(results, report_name="vision_model_comparison"):
    data = []
    for image, image_results in results.items():
        for result in image_results:
            if isinstance(result['groq']['evaluation'], str):
                result['groq']['evaluation'] = json.loads(result['groq']['evaluation'])
            if isinstance(result['openai']['evaluation'], str):
                result['openai']['evaluation'] = json.loads(result['openai']['evaluation'].replace("\n</invoke>", ""))
            # check if any of groq or open ai 'evaluation' is string use json.loads
            data.append(
                {
                    "Image": image,
                    "Prompt": result["prompt"],
                    "Groq_Response": result["groq"]["response"],
                    "Groq_Time": result["groq"]["time"],
                    "Groq_Accuracy": result["groq"]["evaluation"]["accuracy"],
                    "Groq_Completeness": result["groq"]["evaluation"]["completeness"],
                    "Groq_Relevance": result["groq"]["evaluation"]["relevance"],
                    "Groq_Insight": result["groq"]["evaluation"]["insight"],
                    "Groq_Total": result["groq"]["evaluation"]["total"],
                    "Groq_Comments": result["groq"]["evaluation"]["comments"],
                    "OpenAI_Response": result["openai"]["response"],
                    "OpenAI_Time": result["openai"]["time"],
                    "OpenAI_Accuracy": result["openai"]["evaluation"]["accuracy"],
                    "OpenAI_Completeness": result["openai"]["evaluation"][
                        "completeness"
                    ],
                    "OpenAI_Relevance": result["openai"]["evaluation"]["relevance"],
                    "OpenAI_Insight": result["openai"]["evaluation"]["insight"],
                    "OpenAI_Total": result["openai"]["evaluation"]["total"],
                    "OpenAI_Comments": result["openai"]["evaluation"]["comments"],
                }
            )

    df = pd.DataFrame(data)
    os.makedirs("report", exist_ok=True)
    df.to_csv(__location__ + f"/report/{report_name}.csv", index=False)
    with open(__location__ + f"/report/{report_name}.json", "w") as f:
        json.dump(results, f)
    print("Results saved.")


if __name__ == "__main__":
    # Example usage
    image_data = {
        "assets/complex_street_small_1.png": {
            "ground_truth": "Urban intersection with 1 cars, 1 unit of traffic lights, and a prominent red brick building",
            "prompts": [
                "Describe the scene in detail.",
                "How many distinct vehicles and traffic light units can you see in the image?",
                "What's the most prominent building in the image?",
            ],
        },
        "assets/abstract_painting.png": {
            "ground_truth": "Abstract painting with swirling patterns in blue, red, and yellow, resembling a turbulent sky",
            "prompts": [
                "Describe the colors and patterns in this image.",
                "What emotions does this painting evoke?",
                "Can you identify any hidden shapes or figures in the painting?",
            ],
        },
        "assets/receipt.jpeg": { 
            "ground_truth": "Sales receipt from Cider Cellar showing 2 items (Bulmers Original Bottle and Bulmers Pear Bottle) each priced at £4.00, with discounts applied, resulting in a total of £4.50",
            "prompts": [
                "What is the name of the business on this receipt?",
                "List the items purchased and their individual prices.",
                "What is the total amount paid, and were any discounts applied?",
            ],
        },
        # "assets/technical_diagram.png": {
        #     "ground_truth": "Flowchart of software development lifecycle with 6 main stages: Planning, Analysis, Design, Implementation, Testing, and Maintenance",
        #     "prompts": [
        #         "What process does this diagram represent?",
        #         "How many main stages are there in this process?",
        #         "What's the relationship between the different elements in the diagram?"
        #     ]
        # },
        "assets/natural_landscape.png": {
            "ground_truth": "Forest scene with 3 hidden animals: a deer, an owl, and a fox. Dense foliage with a small stream running through.",
            "prompts": [
                "Describe the landscape in this image.",
                "Can you spot any hidden animals? If so, what are they?",
                "What time of day does this scene appear to be set in, and why?",
            ],
        },
    }

    report_name = "vision_model_comparison"
    results = {}

    total = 12
    ix = 1
    for image, data in image_data.items():
        image_results = []
        for prompt in data["prompts"]:
            print(
                f"[LOG] Process case {ix}/{total}, Image: {image}, Prompt: {prompt[:50]}..."
            )
            result = compare_models(image, prompt, data["ground_truth"])
            image_results.append(result)
            ix += 1
        results[image] = image_results

    save_results_to_csv(results, report_name=report_name)

    df = load_data(report_name=report_name)

    # Empty all images from the report folder
    for file in os.listdir(__location__ + "/report/images"):
        os.remove(os.path.join(__location__ + "/report/images", file))

    speed_comparison_bar(df)
    speed_comparison_box(df)
    speed_scatter(df)
    performance_radar(df)
    performance_grouped_bar(df)
    performance_vs_speed_scatter(df)
    improved_overall_comparison(df)
    # efficiency_scatter(df)
    # model_size_performance_bar()
    time_savings_area(df)

    generate_markdown_report(df, report_name=report_name)

    print("Analysis complete. Report and visualizations saved in the 'report' folder.")
