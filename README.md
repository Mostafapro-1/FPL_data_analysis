Project Overview This project focuses on data analytics for the Fantasy Premier League (FPL) 24/25 season, with detailed insights generated up to week 29.

Data Acquisition and Preparation The dataset is sourced directly from the official FPL API via "https://fantasy.premierleague.com/api/bootstrap-static/". which provides player and team statistics necessary for in-depth analysis.

Data Cleaning and Transformation: To ensure the dataset is both relevant and accurate, the following cleaning steps were implemented: Filtering: Excluded managers from the dataset to focus solely on player data. Mapping: Created mappings to convert team IDs into team names and to translate numerical position codes into standard abbreviations (e.g., GK, DEF, MID, FWD). Conditional Adjustments: Set the values for 'Clean Sheets' and 'Goals Conceded' to zero for positions where these statistics are not applicable. Feature Engineering: Constructed a new 'full name' column by combining the players' first and second names, enhancing data readability. Metric Calculation: Introduced a performance metric, 'ppm' (points per million), calculated by dividing total points by the player's cost. This metric is instrumental in identifying high-value players. 

Data Structuring: The dataset is sorted in descending order based on the 'ppm' metric, highlighting the most efficient players. Additionally, key columns were reordered and put at the beginning, then Two empty columns were inserted as a visual separator between the primary columns used and the remaining columns, ensuring clarity and ease of navigation. 

Exporting the Dataset: Once cleaned and organized, the dataset is exported as a CSV file with UTF-8 encoding. This format ensures compatibility and ease of use for subsequent data analysis tasks.
