import pandas as pd
from pathlib import Path

# Load full datasets
notes = pd.read_parquet("notes_full.parquet")
ratings = pd.read_csv("communitynotes/scoring/src/data/filtered_ratings.tsv", sep='\t')
note_status = pd.read_csv("communitynotes/scoring/src/data/noteStatusHistory-00000.tsv", sep="\t")
user_enroll = pd.read_csv("communitynotes/scoring/src/data/userEnrollment-00000.tsv", sep="\t")
original_notes = pd.read_csv('communitynotes/scoring/src/data/filtered_notes.tsv', sep='\t')
expected_columns = original_notes.columns.tolist()


for topic in sorted(notes['topic'].unique()):
    topic_name = topic.replace('&', 'and').replace(' ', '_')
    topic_dir = Path(f"mf_inputs/topic_{topic_name}")
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    topic_notes = notes[notes['topic'] == topic].copy()

    note_ids = set(topic_notes['noteId'])
    topic_ratings = ratings[ratings['noteId'].isin(note_ids)].copy()
    
    user_counts = topic_ratings['raterParticipantId'].value_counts()
    active_users = set(user_counts[user_counts >= 2].index)
    topic_ratings = topic_ratings[topic_ratings['raterParticipantId'].isin(active_users)]
    
    note_ids = set(topic_ratings['noteId'].unique())
    topic_notes = topic_notes[topic_notes['noteId'].isin(note_ids)]
    
    # SANITY CHECK
    if len(topic_notes) < 5 or len(topic_ratings) < 10:
        continue
    
    topic_notes['summary'] = topic_notes['summary_x']
    topic_notes_filtered = topic_notes[expected_columns]
    topic_status = note_status[note_status['noteId'].isin(note_ids)].copy()
    topic_enrollment = user_enroll.copy()

    #SANITY CHECK
    if topic_notes_filtered.empty or topic_ratings.empty or topic_status.empty:
        print(f"{topic}: input files are empty")
        continue
    
    topic_notes_filtered.to_csv(topic_dir / 'notes-00000.tsv', sep='\t', index=False)
    topic_ratings.to_csv(topic_dir / 'ratings-00000.tsv', sep='\t', index=False)
    topic_status.to_csv(topic_dir / 'noteStatusHistory-00000.tsv', sep='\t', index=False)
    topic_enrollment.to_csv(topic_dir / 'userEnrollment-00000.tsv', sep='\t', index=False)
    
    print(f"{topic}: {len(topic_notes)} notes {len(topic_ratings)} ratings {len(topic_status)} status")

