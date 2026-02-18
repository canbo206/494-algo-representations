#!/usr/bin/env python3

"""Invoke Community Notes scoring and user contribution algorithms.

Example Usage:
  # If there is only one rating file, pass the file path to the --ratings flag.
  python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes-00000.tsv \
    --ratings data/ratings-00000.tsv \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data

  # If there are multiple rating files, move them to a directory, 
  # and pass the directory path to the --ratings flag.
  python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes-00000.tsv \
    --ratings data/ratings \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data
"""

import logging

from scoring.runner import main


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  main()


"""If there is only one rating file, pass the file path to the --ratings flag.
  # python3 communitynotes/scoring/src/main.py \
  --enrollment mf_inputs/topic_news_and_social_concern/userEnrollment-00000.tsv \
  --notes mf_inputs/topic_news_and_social_concern/notes-00000.tsv \
  --ratings mf_inputs/topic_news_and_social_concern/ratings-00000.tsv \
  --status mf_inputs/topic_news_and_social_concern/noteStatusHistory-00000.tsv \
  --outdir topic_outputs/topic_news_and_social_concern
"""

"""Warning: No scored notes available for author helpfulness calculation
Warning: No author counts - returning empty helpfulness scores
Warning: No helpfulness scores - returning all ratings
Warning: No scored notes available for author helpfulness calculation
Warning: No author counts - returning empty helpfulness scores
Warning: No helpfulness scores - returning all ratings
Warning: NaN loss detected in diligence model training - skipping this round
Warning: NaN loss detected in diligence model training - skipping this round
"""