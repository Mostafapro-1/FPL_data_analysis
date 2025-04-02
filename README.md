Project Overview:

This project focuses on data analytics for the Fantasy Premier League (FPL) 24/25 season, with detailed insights generated up to week 29.



Data Acquisition and Preparation:

The dataset is sourced directly from the official FPL API via "https://fantasy.premierleague.com/api/bootstrap-static/". which provides player and team statistics necessary for in-depth analysis.



Data Cleaning and Transformation:

To ensure the dataset is both relevant and accurate, the following cleaning steps were implemented: 

1.Filtering: Excluded managers from the dataset to focus solely on player data. 

2.Mapping: Created mappings to convert team IDs into team names and to translate numerical position codes into standard abbreviations (e.g., GK, DEF, MID, FWD). 

3.Conditional Adjustments: Set the values for 'Clean Sheets' and 'Goals Conceded' to zero for positions where these statistics are not applicable. 

4.Feature Engineering: Constructed a new 'full name' column by combining the players' first and second names, enhancing data readability. 


Data Structuring: 
The dataset is sorted in descending order based on the 'ppm' metric, highlighting the most efficient players. Additionally, key columns were reordered and put at the beginning of the csv file, then Two empty columns were inserted as a visual separator between the primary columns used and the remaining columns, ensuring clarity and ease of navigation. 



Exporting the Dataset: Once cleaned and organized, the dataset is exported as a CSV file with UTF-8 encoding. This format ensures compatibility and ease of use for subsequent data analysis tasks.


then,I collaborated with AI tools like ChatGPT/Deepseek to bridge my sports passion with technical execution. Here’s how I did it:

Key Features:

 Points per Million (PPM) Analysis: Identified cost-effective players using Python (Pandas/NumPy).

 Interactive Dashboards: Streamlit tables and Plotly charts to explore performance metrics.

 AI-Augmented Development: Guided ChatGPT with targeted prompts to generate, test, and refine code iteratively.

My Process:

 Problem Decomposition: Broke down goals into technical tasks (e.g., “How to filter players by position in Streamlit?”).

 AI Collaboration: Used ChatGPT to draft code snippets, then validated and debugged outputs.

 Iterative Refinement: Tweaked prompts to optimize visuals (e.g., adjusting Plotly bubble charts for clarity).

 Ownership: Every line of code was reviewed, tested, and adapted to fit the project’s needs.



Tech Stack:

Python | Streamlit | Plotly

AI-Assisted Development | Data Storytelling

Proud Moment:
Delivering a polished tool despite starting with limited coding experience – proof that curiosity + AI collaboration can unlock rapid growth!

Looking Ahead:
I’m eager to bring this same analytical rigor to the sports analytics field, whether it’s optimizing team strategies, scouting talent, or enhancing fan engagement.
