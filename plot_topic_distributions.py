import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
import warnings
warnings.filterwarnings('ignore')

def load_topic_results():

    all_results = {}
    all_summaries = []
    
    outputs_dir = Path('topic_mf_outputs')
    
    for topic_dir in outputs_dir.glob('topic_*'):
        topic_name = topic_dir.name.replace('topic_', '')
        
        summary_file = topic_dir / 'summary.tsv'
        if summary_file.exists():
            try:
                summary = pd.read_csv(summary_file, sep='\t')
                all_summaries.append(summary.iloc[0].to_dict())

                note_factors_file = topic_dir / 'note_factors.tsv'
                if note_factors_file.exists():
                    note_factors = pd.read_csv(note_factors_file, sep='\t')
                    all_results[topic_name] = note_factors
                    
                print(f"Loaded {topic_name}: {summary.iloc[0]['n_notes']} notes, {summary.iloc[0]['n_ratings']} ratings")
            except Exception as e:
                print(f"{topic_name}: {e}")
        else:
            print(f"No summary for {topic_name}")
    
    return all_results, all_summaries

def analyze_topic_distributions(all_results, summaries_df):
    if len(all_results) < 2:
        print("Need at least 2 topics for comparison")
        return
    
    all_intercepts = []
    all_factors1 = []
    all_factors2 = []
    
    for data in all_results.values():
        all_intercepts.extend(data['noteIntercept'].values)
        all_factors1.extend(data['noteFactor1'].values)
        all_factors2.extend(data['noteFactor2'].values)
    
    global_intercepts = np.array(all_intercepts)
    global_factors1 = np.array(all_factors1)
    global_factors2 = np.array(all_factors2)
    
    distances = []
    
    for topic, data in all_results.items():
        topic_intercepts = data['noteIntercept'].values
        topic_factors1 = data['noteFactor1'].values
        topic_factors2 = data['noteFactor2'].values
        
        # Wasserstein distances
        wasserstein_intercept = wasserstein_distance(topic_intercepts, global_intercepts)
        wasserstein_factor1 = wasserstein_distance(topic_factors1, global_factors1)
        wasserstein_factor2 = wasserstein_distance(topic_factors2, global_factors2)
        
        # KS statistics
        ks_intercept = ks_2samp(topic_intercepts, global_intercepts).statistic
        ks_factor1 = ks_2samp(topic_factors1, global_factors1).statistic
        ks_factor2 = ks_2samp(topic_factors2, global_factors2).statistic
        
        distances.append({'topic': topic,'n_notes': len(data),'wasserstein_intercept': wasserstein_intercept,
            'wasserstein_factor1': wasserstein_factor1,'wasserstein_factor2': wasserstein_factor2,
            'ks_intercept': ks_intercept,'ks_factor1': ks_factor1,'ks_factor2': ks_factor2
        })
    
    distances_df = pd.DataFrame(distances)
    
    create_distribution_plots(all_results, summaries_df, distances_df)
    
    plots_dir = Path('topic_distribution_analysis')
    plots_dir.mkdir(exist_ok=True)
    
    summaries_df.to_csv(plots_dir / 'topic_summaries.csv', index=False)
    distances_df.to_csv(plots_dir / 'topic_distances.csv', index=False)
    
    print_distribution_analysis(summaries_df, distances_df)
    
    return distances_df

def create_distribution_plots(all_results, summaries_df, distances_df):
    plt.style.use('default')
    sns.set_palette("husl")
    
    plots_dir = Path('topic_distribution_analysis')
    plots_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Topic Aggregate Statistics Comparison', fontsize=16)
    
    summaries_sorted = summaries_df.sort_values('n_notes', ascending=True)
    topics_list = summaries_sorted['topic'].tolist()
    
    ax = axes[0, 0]
    bars = ax.bar(range(len(summaries_sorted)), summaries_sorted['n_notes'], color='skyblue', alpha=0.7)
    ax.set_xlabel('Topic (sorted by size)')
    ax.set_ylabel('Number of Notes')
    ax.set_title('Topic Sizes (Number of Notes)')
    ax.set_xticks(range(len(summaries_sorted)))
    ax.set_xticklabels([t.replace('_', ' ')[:15] + '...' if len(t) > 15 else t.replace('_', ' ') for t in topics_list], rotation=45, ha='right')
    
    #Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    #Number of ratings
    ax = axes[0, 1]
    bars = ax.bar(range(len(summaries_sorted)), summaries_sorted['n_ratings'], color='lightcoral', alpha=0.7)
    ax.set_xlabel('Topic (sorted by size)')
    ax.set_ylabel('Number of Meaningful Ratings')
    ax.set_title('Topic Engagement (Meaningful Ratings)')
    ax.set_xticks(range(len(summaries_sorted)))
    ax.set_xticklabels([t.replace('_', ' ')[:15] + '...' if len(t) > 15 else t.replace('_', ' ') for t in topics_list], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Mean intercepts by topic
    ax = axes[1, 0]
    bars = ax.bar(range(len(summaries_sorted)), summaries_sorted['mean_note_intercept'], color='lightgreen', alpha=0.7)
    ax.set_xlabel('Topic (sorted by size)')
    ax.set_ylabel('Mean Note Intercept')
    ax.set_title('Mean Note Intercepts by Topic')
    ax.set_xticks(range(len(summaries_sorted)))
    ax.set_xticklabels([t.replace('_', ' ')[:15] + '...' if len(t) > 15 else t.replace('_', ' ') for t in topics_list], rotation=45, ha='right')
    
    # Mean factor by topic
    ax = axes[1, 1]
    bars = ax.bar(range(len(summaries_sorted)), summaries_sorted['mean_note_factor1'], color='plum', alpha=0.7)
    ax.set_xlabel('Topic (sorted by size)')
    ax.set_ylabel('Mean Note Factor 1')
    ax.set_title('Mean Note Factor 1 by Topic')
    ax.set_xticks(range(len(summaries_sorted)))
    ax.set_xticklabels([t.replace('_', ' ')[:15] + '...' if len(t) > 15 else t.replace('_', ' ') for t in topics_list], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'aggregate_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    #Distribution plots for note intercepts and factors
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Note Distributions by Topic', fontsize=16)
    
    topics = list(all_results.keys())
    colors = sns.color_palette("husl", len(topics))
    
    # Histogram of intercepts
    ax = axes[0, 0]
    for i, (topic, data) in enumerate(all_results.items()):
        if len(data) > 0:
            ax.hist(data['noteIntercept'], alpha=0.7, label=topic.replace('_', ' '), bins=min(10, len(data)), color=colors[i])
    ax.set_xlabel('Note Intercept')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Note Intercepts')
    ax.legend()
    
    # Box plot of intercepts
    ax = axes[0, 1]
    intercept_data = []
    intercept_labels = []
    for topic, data in all_results.items():
        if len(data) > 0:
            intercept_data.extend(data['noteIntercept'].values)
            intercept_labels.extend([topic.replace('_', ' ')] * len(data))
    
    if intercept_data:
        intercept_df = pd.DataFrame({'Intercept': intercept_data, 'Topic': intercept_labels})
        sns.boxplot(data=intercept_df, x='Topic', y='Intercept', ax=ax)
        ax.set_title('Box Plot of Note Intercepts by Topic')
        ax.tick_params(axis='x', rotation=45)
    

    ax = axes[1, 0]
    for i, (topic, data) in enumerate(all_results.items()):
        if len(data) > 0:
            ax.hist(data['noteFactor1'], alpha=0.7, label=topic.replace('_', ' '), bins=min(10, len(data)), color=colors[i])
    ax.set_xlabel('Note Factor 1')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Note Factor 1')
    ax.legend()
    
    ax = axes[1, 1]
    factor1_data = []
    factor1_labels = []
    for topic, data in all_results.items():
        if len(data) > 0:
            factor1_data.extend(data['noteFactor1'].values)
            factor1_labels.extend([topic.replace('_', ' ')] * len(data))
    
    if factor1_data:
        factor1_df = pd.DataFrame({'Factor1': factor1_data, 'Topic': factor1_labels})
        sns.boxplot(data=factor1_df, x='Topic', y='Factor1', ax=ax)
        ax.set_title('Box Plot of Note Factor 1 by Topic')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Topic Distance from Global Distribution', fontsize=16)
    
    distances_sorted = distances_df.sort_values('topic')
    x_pos = np.arange(len(distances_sorted))
    topic_names = [t.replace('_', ' ') for t in distances_sorted['topic'].tolist()]
    
    # Wasserstein distances
    ax = axes[0]
    ax.bar(x_pos, distances_sorted['wasserstein_intercept'], alpha=0.7, label='Intercept', color='blue')
    ax.bar(x_pos + 0.25, distances_sorted['wasserstein_factor1'], alpha=0.7, label='Factor 1', color='red')
    ax.bar(x_pos + 0.5, distances_sorted['wasserstein_factor2'], alpha=0.7, label='Factor 2', color='green')
    ax.set_xlabel('Topic')
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Wasserstein Distance from Global Distribution')
    ax.set_xticks(x_pos + 0.25)
    ax.set_xticklabels(topic_names, rotation=45, ha='right')
    ax.legend()
    
    # KS statistics
    ax = axes[1]
    ax.bar(x_pos, distances_sorted['ks_intercept'], alpha=0.7, label='Intercept', color='blue')
    ax.bar(x_pos + 0.25, distances_sorted['ks_factor1'], alpha=0.7, label='Factor 1', color='red')
    ax.bar(x_pos + 0.5, distances_sorted['ks_factor2'], alpha=0.7, label='Factor 2', color='green')
    ax.set_xlabel('Topic')
    ax.set_ylabel('KS Statistic')
    ax.set_title('KS Statistic from Global Distribution')
    ax.set_xticks(x_pos + 0.25)
    ax.set_xticklabels(topic_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_distribution_analysis(summaries_df, distances_df):
    print(f"Topics by Size:")
    top_5_by_size = summaries_df.nlargest(5, 'n_notes')
    for _, row in top_5_by_size.iterrows():
        print(f"  {row['topic'].replace('_', ' ')}: {row['n_notes']} notes ({row['n_ratings']} ratings)")
    
    print(f"Most Different Topics (by Wasserstein distance):")
    most_different_intercept = distances_df.loc[distances_df['wasserstein_intercept'].idxmax()]
    most_different_factor1 = distances_df.loc[distances_df['wasserstein_factor1'].idxmax()]
    print(f"By intercept: {most_different_intercept['topic'].replace('_', ' ')} ({most_different_intercept['wasserstein_intercept']:.3f})")
    print(f"By factor 1: {most_different_factor1['topic'].replace('_', ' ')} ({most_different_factor1['wasserstein_factor1']:.3f})")
    
    print(f"Most Engaging Topics (by ratings per note):")
    summaries_df['ratings_per_note'] = summaries_df['n_ratings'] / summaries_df['n_notes']
    top_5_engagement = summaries_df.nlargest(5, 'ratings_per_note')
    for _, row in top_5_engagement.iterrows():
        print(f"  {row['topic'].replace('_', ' ')}: {row['ratings_per_note']:.1f} ratings/note")
    
    print(f"Topic Agreeability (based on mean intercept):")
    most_agreeable = summaries_df.loc[summaries_df['mean_note_intercept'].idxmax()]
    least_agreeable = summaries_df.loc[summaries_df['mean_note_intercept'].idxmin()]
    print(f"Most agreeable: {most_agreeable['topic'].replace('_', ' ')} ({most_agreeable['mean_note_intercept']:.3f})")
    print(f"Least agreeable: {least_agreeable['topic'].replace('_', ' ')} ({least_agreeable['mean_note_intercept']:.3f})")

def main():

    all_results, all_summaries = load_topic_results()
    
    if len(all_results) == 0:
        return
    
    print(f"Found results for {len(all_results)} topics")
    
    summaries_df = pd.DataFrame(all_summaries)
    
    distances_df = analyze_topic_distributions(all_results, summaries_df)
    
    print(f"\nDistribution analysis complete!")
    print(f"Results saved to topic_distribution_analysis/")

if __name__ == "__main__":
    main()
