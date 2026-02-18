import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def run_simplified_mf(topic_name):
    
    input_dir = Path(f'mf_inputs/topic_{topic_name}')
    output_dir = Path(f'topic_mf_outputs/topic_{topic_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return None, None
    
    try:
        ratings = pd.read_csv(input_dir / 'ratings-00000.tsv', sep='\t')
        notes = pd.read_csv(input_dir / 'notes-00000.tsv', sep='\t')
        
        print(f"Loaded {len(ratings):,} ratings and {len(notes):,} notes")
    
        ratings['rating_value'] = ratings['helpful'] - ratings['notHelpful']
        meaningful_ratings = ratings[ratings['rating_value'] != 0].copy()
        
        if len(meaningful_ratings) == 0:
            print("No meaningful ratings found!")
            return None, None
        
        print(f"{len(meaningful_ratings):,} meaningful ratings")
        

        user_encoder = LabelEncoder()
        note_encoder = LabelEncoder()
        user_ids = user_encoder.fit_transform(meaningful_ratings['raterParticipantId'])
        note_ids = note_encoder.fit_transform(meaningful_ratings['noteId'])
        
        n_users = len(user_encoder.classes_)
        n_notes = len(note_encoder.classes_)
        
        user_note_matrix = np.zeros((n_users, n_notes))
        for user_id, note_id, rating in zip(user_ids, note_ids, meaningful_ratings['rating_value'].abs()):
            user_note_matrix[user_id, note_id] = rating

        n_factors = min(10, min(n_users, n_notes) // 2)
        if n_factors < 2:
            n_factors = 2
        
        nmf = NMF(n_components=n_factors, random_state=42, max_iter=200)
        
        try:
            W = nmf.fit_transform(user_note_matrix)
            H = nmf.components_

            note_factors = H.T
            
            # Calculate note intercepts (average rating per note)
            note_intercepts = []
            for note_idx in range(n_notes):
                note_ratings = user_note_matrix[:, note_idx]
                note_ratings = note_ratings[note_ratings != 0]
                if len(note_ratings) > 0:
                    note_intercepts.append(np.mean(note_ratings))
                else:
                    note_intercepts.append(0.0)
            
            note_intercepts = np.array(note_intercepts)
            
            elapsed = time.time() - start_time
            print(f"NMF completed in {elapsed:.3f} seconds")
            
            # Create output dataframes
            note_results = pd.DataFrame({
                'noteId': note_encoder.inverse_transform(range(n_notes)),
                'noteIntercept': note_intercepts,
                'noteFactor1': note_factors[:, 0],
                'noteFactor2': note_factors[:, 1] if note_factors.shape[1] > 1 else 0.0
            })
            
            user_results = pd.DataFrame({
                'raterParticipantId': user_encoder.inverse_transform(range(n_users)),
                'userFactor1': W[:, 0],
                'userFactor2': W[:, 1] if W.shape[1] > 1 else 0.0
            })
            
            # Save outputs
            note_results.to_csv(output_dir / 'note_factors.tsv', sep='\t', index=False)
            user_results.to_csv(output_dir / 'user_factors.tsv', sep='\t', index=False)
            
            summary = {
                'topic': topic_name,'n_notes': n_notes,'n_users': n_users,'n_ratings': len(meaningful_ratings),
                'n_factors': n_factors,'runtime_seconds': elapsed,'mean_note_intercept': np.mean(note_intercepts),
                'std_note_intercept': np.std(note_intercepts),'mean_note_factor1': np.mean(note_factors[:, 0]),
                'std_note_factor1': np.std(note_factors[:, 0])
            }
            
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(output_dir / 'summary.tsv', sep='\t', index=False)
            
            print(f"Results saved to: {output_dir}")
            print(f"Mean intercept: {np.mean(note_intercepts):.3f} (±{np.std(note_intercepts):.3f})")
            print(f"Mean factor 1: {np.mean(note_factors[:, 0]):.3f} (±{np.std(note_factors[:, 0]):.3f})")
            
            return note_results, summary
            
        except Exception as e:
            return None, None
            
    except Exception as e:
        return None, None

def main():
    # Get all available topics
    mf_inputs_dir = Path('mf_inputs')
    topics = []
    
    for topic_dir in mf_inputs_dir.glob('topic_*'):
        topic_name = topic_dir.name.replace('topic_', '')
        topics.append(topic_name)
    
    print(f"Found {len(topics)} topics: {', '.join(topics[:5])}{'...' if len(topics) > 5 else ''}")
    
    # Sort topics by expected size 
    print("Sorting topics by size (smallest first)")
    topic_sizes = {}
    for topic in topics:
        input_dir = Path(f'mf_inputs/topic_{topic}')
        try:
            ratings = pd.read_csv(input_dir / 'ratings-00000.tsv', sep='\t')
            topic_sizes[topic] = len(ratings)
        except:
            topic_sizes[topic] = 0
    
    topics_sorted = sorted(topics, key=lambda x: topic_sizes.get(x, 0))
    
    print(f"Topic sizes (ratings): {[(t, topic_sizes[t]) for t in topics_sorted[:5]]}")
    
    all_results = {}
    all_summaries = []
    total_start_time = time.time()
    individual_runtimes = []
    
    # Process each topic (smallest first)
    for i, topic in enumerate(topics_sorted, 1):
        print(f"\nProgress: {i}/{len(topics_sorted)} - {topic}")
        
        topic_start_time = time.time()
        note_results, summary = run_simplified_mf(topic)
        topic_elapsed = time.time() - topic_start_time
        individual_runtimes.append(topic_elapsed)
        
        if note_results is not None:
            all_results[topic] = note_results
            all_summaries.append(summary)
            print(f"Topic runtime: {topic_elapsed:.3f}s")
        else:
            print(f"Skipped {topic}")
 
    if len(all_results) == 0:
        return
 
    summaries_df = pd.DataFrame(all_summaries)
    summaries_df.to_csv('topic_mf_outputs/all_topic_summaries.csv', index=False)
    


if __name__ == "__main__":
    main()
