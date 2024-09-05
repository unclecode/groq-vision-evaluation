import os
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure required directories exist
os.makedirs(__location__ + '/report/images', exist_ok=True)

def load_data(report_name='vision_model_comparison'):
    return pd.read_csv(__location__ + f'/report/{report_name}.csv')

def speed_comparison_bar(df):
    plt.figure(figsize=(15, 8))
    x = range(len(df))
    width = 0.35
    plt.bar(x, df['Groq_Time'], width, label='Groq', color='blue', alpha=0.7)
    plt.bar([i + width for i in x], df['OpenAI_Time'], width, label='OpenAI', color='green', alpha=0.7)
    plt.xlabel('Image-Prompt Combination')
    plt.ylabel('Response Time (s)')
    plt.title('Speed Comparison: Groq vs OpenAI')
    plt.legend()
    plt.xticks([i + width/2 for i in x], df.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(__location__ + '/report/images/speed_comparison_bar.png')
    plt.close()

def speed_comparison_box(df):
    plt.figure(figsize=(10, 6))
    data = [df['Groq_Time'], df['OpenAI_Time']]
    plt.boxplot(data, labels=['Groq', 'OpenAI'])
    plt.ylabel('Response Time (s)')
    plt.title('Distribution of Response Times')
    plt.savefig(__location__ + '/report/images/speed_comparison_box.png')
    plt.close()

def speed_scatter(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Groq_Time'], df['OpenAI_Time'])
    plt.xlabel('Groq Time (s)')
    plt.ylabel('OpenAI Time (s)')
    plt.title('Groq Time vs OpenAI Time')
    max_time = max(df['Groq_Time'].max(), df['OpenAI_Time'].max())
    plt.plot([0, max_time], [0, max_time], 'r--')  # Diagonal line
    plt.savefig(__location__ + '/report/images/speed_scatter.png')
    plt.close()

def performance_radar(df):
    metrics = ['Accuracy', 'Completeness', 'Relevance', 'Insight']
    
    fig = go.Figure()

    for index, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[f'Groq_{m}'] for m in metrics],
            theta=metrics,
            fill='toself',
            name=f'Groq - {index}'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[row[f'OpenAI_{m}'] for m in metrics],
            theta=metrics,
            fill='toself',
            name=f'OpenAI - {index}'
        ))

    # Add legend to figure
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True
    )
    
    fig.write_image(__location__ + "/report/images/performance_radar.png")

def performance_grouped_bar(df):
    metrics = ['Accuracy', 'Completeness', 'Relevance', 'Insight']
    
    groq_means = [df[f'Groq_{m}'].mean() for m in metrics]
    openai_means = [df[f'OpenAI_{m}'].mean() for m in metrics]

    fig = go.Figure(data=[
        go.Bar(name='Groq', x=metrics, y=groq_means, marker_color='blue'),
        go.Bar(name='OpenAI', x=metrics, y=openai_means, marker_color='green')
    ])

    fig.update_layout(
        barmode='group',
        title='Performance Metrics Comparison',
        xaxis_title='Metrics',
        yaxis_title='Average Score',
        legend_title='Model',
        font=dict(size=12),
        yaxis=dict(range=[0, 10])  # Set y-axis range from 0 to 10
    )

    fig.write_image(__location__ + "/report/images/performance_grouped_bar.png")

def performance_vs_speed_scatter(df):
    plt.figure(figsize=(12, 8))
    
    # Plot Groq data
    plt.scatter(df['Groq_Total'], df['Groq_Time'], 
                color='orange', label='Groq', alpha=0.7)
    
    # Plot OpenAI data
    plt.scatter(df['OpenAI_Total'], df['OpenAI_Time'], 
                color='blue', label='OpenAI', alpha=0.7)
    
    plt.xlabel('Total Score (Performance/Accuracy)')
    plt.ylabel('Response Time (seconds)')
    plt.title('Performance vs Speed: Groq vs OpenAI')
    plt.legend()
    
    # Set y-axis to logarithmic scale to better show the time difference
    plt.yscale('log')
    
    # Set x-axis range from 31 to 40
    plt.xlim(31, 40)
    
    # Add a grid for better readability
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(__location__ + '/report/images/performance_vs_speed_scatter.png')
    plt.close()


def improved_overall_comparison(df):
    # Calculate mean scores for each metric
    metrics = ['Time', 'Accuracy', 'Completeness', 'Relevance', 'Insight', 'Total']
    groq_means = [df[f'Groq_{m}'].mean() for m in metrics]
    openai_means = [df[f'OpenAI_{m}'].mean() for m in metrics]

    # Create a new dataframe for the heatmap
    heatmap_data = pd.DataFrame({
        'Metric': metrics,
        'Groq': groq_means,
        'OpenAI': openai_means
    })
    heatmap_data = heatmap_data.set_index('Metric')

    # Calculate the difference (OpenAI - Groq)
    heatmap_data['Difference'] = heatmap_data['OpenAI'] - heatmap_data['Groq']

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Grouped bar chart
    x = range(len(metrics))
    width = 0.35
    ax1.bar([i - width/2 for i in x], groq_means, width, label='Groq', color='blue', alpha=0.7)
    ax1.bar([i + width/2 for i in x], openai_means, width, label='OpenAI', color='green', alpha=0.7)

    ax1.set_ylabel('Scores')
    ax1.set_title('Overall Comparison: Groq vs OpenAI')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()

    # Add value labels on the bars
    for i, v in enumerate(groq_means):
        ax1.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(openai_means):
        ax1.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')

    # Heatmap for the difference
    sns.heatmap(heatmap_data[['Difference']], ax=ax2, cmap='RdYlGn', center=0, annot=True, fmt='.2f')
    ax2.set_title('Difference (OpenAI - Groq)')

    plt.tight_layout()
    plt.savefig(__location__ + "/report/images/overall_comparison_parallel.png")
    plt.close()


def efficiency_scatter(df):
    fig = px.scatter(df, x='Groq_Time', y='Groq_Total', color='Image',
                     hover_data=['Prompt'])
    fig.add_trace(px.scatter(df, x='OpenAI_Time', y='OpenAI_Total', color='Image',
                             hover_data=['Prompt']).data[0])
    
    for i in range(len(df)):
        fig.add_trace(go.Scatter(x=[df['Groq_Time'][i], df['OpenAI_Time'][i]],
                                 y=[df['Groq_Total'][i], df['OpenAI_Total'][i]],
                                 mode='lines',
                                 line=dict(color='gray', width=1),
                                 showlegend=False))

    fig.update_layout(title='Efficiency Comparison')
    fig.write_image(__location__ + "/report/images/efficiency_scatter.png")

def model_size_performance_bar():
    # Assuming model sizes
    model_sizes = {'Groq (LLaVA)': 7, 'OpenAI': 175}  # in billions
    avg_scores = {'Groq (LLaVA)': df['Groq_Total'].mean(), 'OpenAI': df['OpenAI_Total'].mean()}
    std_scores = {'Groq (LLaVA)': df['Groq_Total'].std(), 'OpenAI': df['OpenAI_Total'].std()}

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_sizes.keys(), avg_scores.values())
    ax.errorbar(model_sizes.keys(), avg_scores.values(), yerr=std_scores.values(), fmt='none', capsize=5, color='black')

    ax.set_ylabel('Average Total Score')
    ax.set_title('Model Size vs Performance')

    # Add model size labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{list(model_sizes.values())[i]}B',
                ha='center', va='bottom')

    plt.savefig(__location__ + '/report/images/model_size_performance_bar.png')
    plt.close()

def time_savings_area(df):
    df_sorted = df.sort_values('Groq_Time')
    cumulative_groq = df_sorted['Groq_Time'].cumsum()
    cumulative_openai = df_sorted['OpenAI_Time'].cumsum()

    fig = go.Figure()
    x_values = list(range(len(df)))  # Convert range to a list
    fig.add_trace(go.Scatter(x=x_values, y=cumulative_groq, fill='tozeroy', name='Groq'))
    fig.add_trace(go.Scatter(x=x_values, y=cumulative_openai, fill='tonexty', name='OpenAI'))

    fig.update_layout(title='Cumulative Time Savings', xaxis_title='Number of Queries', yaxis_title='Cumulative Time (s)')
    fig.write_image(__location__ + "/report/images/time_savings_area.png")

def generate_markdown_report(df, report_name='vision_model_comparison'):
    markdown = """# Vision Model Comparison Report

## Overview
This report compares the performance of Groq and OpenAI vision models across various metrics.

## Speed Comparison
![Speed Comparison Bar Chart](images/speed_comparison_bar.png)
![Speed Comparison Box Plot](images/speed_comparison_box.png)
![Speed Scatter Plot](images/speed_scatter.png)

## Performance Metrics
![Performance Radar Chart](images/performance_radar.png)
![Performance Grouped Bar Chart](images/performance_grouped_bar.png)

## Speed vs Performance
![Speed vs Performance Scatter Plot](images/performance_vs_speed_scatter.png)

## Overall Comparison
![Overall Comparison Parallel Coordinates](images/overall_comparison_parallel.png)

## Time Savings
![Time Savings Area Chart](images/time_savings_area.png)

## Summary Statistics
"""
    markdown += df.describe().to_markdown()

    # Create a new summary table
    markdown += "\n\n## Metric Comparison (Mean ± Std)\n\n"
    
    metrics = ['Time', 'Accuracy', 'Completeness', 'Relevance', 'Insight', 'Total']
    summary_data = []

    for metric in metrics:
        groq_mean = df[f'Groq_{metric}'].mean()
        groq_std = df[f'Groq_{metric}'].std()
        openai_mean = df[f'OpenAI_{metric}'].mean()
        openai_std = df[f'OpenAI_{metric}'].std()
        
        summary_data.append({
            'Metric': metric,
            'Groq (Llava 1.5-7b)': f"{groq_mean:.2f} ± {groq_std:.2f}",
            'OpenAI (Gpt-4o-mini)': f"{openai_mean:.2f} ± {openai_std:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Metric')
    markdown += summary_df.to_markdown()

    with open(__location__ + f'/report/{report_name}.md', 'w') as f:
        f.write(markdown)

if __name__ == '__main__':
    # Main execution
    report_name='vision_model_comparison'
    df = load_data(report_name=report_name)

    # Empty all images from the report folder
    for file in os.listdir(__location__ + '/report/images'):
        os.remove(os.path.join(__location__ + '/report/images', file))
        
    speed_comparison_bar(df)
    speed_comparison_box(df)
    speed_scatter(df)
    performance_radar(df)
    performance_grouped_bar(df)
    performance_vs_speed_scatter(df)
    improved_overall_comparison(df)
    # efficiency_scatter(df)
    model_size_performance_bar()
    time_savings_area(df)

    generate_markdown_report(df, report_name=report_name)

    print("Analysis complete. Report and visualizations saved in the 'report' folder.")