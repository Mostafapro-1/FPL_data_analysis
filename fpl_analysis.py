import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px  
import plotly.graph_objects as go


# Set page config
st.set_page_config(page_title="FPL Player Analysis", layout="wide")


# Load and process data
@st.cache_data
def load_data():
    players_df = pd.read_csv('fpl.csv')
        # Add reverse position mapping FIRST
    position_reverse_map = {
        'GK': 1,
        'DEF': 2,
        'MID': 3,
        'FWD': 4
    }
    
    # Auto-update ElementType based on Position column
    if 'position' in players_df.columns:  # Check if position column exists in CSV
        players_df['element_type'] = players_df['position'].map(position_reverse_map)
    else:
        st.error("Position column not found in CSV! Using original element_type values.")

    # Clean column names
    players_df.columns = (
        players_df.columns.str.replace('_', ' ')
        .str.title()
        .str.replace(' ', '')
    )
  
    
    # Set CleanSheets and GoalsConceded for non-defensive positions
    players_df.loc[players_df['Position'].isin(['FWD']), 'CleanSheets'] = 0
    players_df.loc[players_df['Position'].isin(['MID', 'FWD']), 'GoalsConceded'] = 0
    
    # Calculate metrics
    players_df['CurrentPrice'] = players_df['NowCost'] / 10
    players_df['PPM'] = players_df['TotalPoints'] / players_df['CurrentPrice']
    players_df['PPM'] = np.where(players_df['Position'].notna(), 
                               players_df['PPM'].round(1), np.nan)
    
    return players_df.sort_values(by='PPM', ascending=False)

players_df = load_data()

# Streamlit app layout
st.title("Fantasy Premier League Player Analysis")
st.markdown("---")

# Show raw data
with st.expander("View Processed Data"):
    st.dataframe(players_df, height=300, use_container_width=True)

# ---------------------------
# New Tables Section
# ---------------------------

st.markdown("---")
st.header("Key Player Metrics Tables")

# Table 1: Top 10 PPM Players
st.subheader("Top Value Players: Points per Million(PPM)")
top_ppm = players_df[['FullName', 'Team', 'Position', 'CurrentPrice', 
                     'TotalPoints', 'Minutes', 'PPM']].sort_values('PPM', ascending=False)
show_all_ppm = st.checkbox("Show all top PPM players", key="ppm_all")
st.dataframe(
    top_ppm.head(10) if not show_all_ppm else top_ppm,
    column_config={
        "CurrentPrice": st.column_config.NumberColumn(format="£%.1f"),
        "PPM": st.column_config.NumberColumn(format="%.1f")
    },
    hide_index=True,  # <--- ADD THIS LINE
    height=400,
    use_container_width=True
)

# Table 2: Top PPM by Position with Team Filter
st.subheader("Top Value Players: Points per Million(PPM) by Position")

# Create tabs for each position
tab_fwd, tab_mid, tab_def, tab_gk = st.tabs(["Forwards", "Midfielders", "Defenders", "Goalkeepers"])

# Create list of tuples with (tab_object, position_label, position_code)
position_tabs = [
    (tab_fwd, "Forwards", "FWD"),
    (tab_mid, "Midfielders", "MID"),
    (tab_def, "Defenders", "DEF"),
    (tab_gk, "Goalkeepers", "GK")
]

for tab, label, pos in position_tabs:
    with tab:  # Use the tab OBJECT here
        # Get players for this position
        pos_df = players_df[players_df['Position'] == pos]
        
        # Team selection dropdown
        team_list = ["All Teams"] + sorted(pos_df['Team'].unique().tolist())
        selected_team = st.selectbox(
            f"Select Team ({pos})", 
            team_list,
            key=f"team_select_{pos}"
        )
        
        # Apply both position and team filters
        if selected_team != "All Teams":
            pos_df = pos_df[pos_df['Team'] == selected_team]
        
        # Sort and display
        pos_top = pos_df[['FullName', 'Team', 'CurrentPrice', 
                         'TotalPoints', 'Minutes', 'PPM']].sort_values('PPM', ascending=False)
        
        show_all = st.checkbox(f"Show all {pos} players", key=f"all_{pos}")
        
        st.dataframe(
            pos_top if show_all else pos_top.head(10),
            column_config={
                "CurrentPrice": st.column_config.NumberColumn(format="£%.1f"),
                "PPM": st.column_config.NumberColumn(format="%.1f")
            },
            hide_index=True,
            height=400,
            use_container_width=True
        )

# Table 3: Risk Players
st.subheader("High Risk/Reward Players")
# Add a note under the title
st.markdown(
    "These players are high-risk/high-reward options in FPL. " "They have a great PPM (Points per Million) but are tricky, "
    "as they have 25% less playtime. This means they could either " "deliver big returns or fail to earn you points."
)

risk_players = players_df[
    (players_df['Minutes'] <= 1960) & 
    (players_df['PPM'].notna())
].sort_values('PPM', ascending=False)

risk_top = risk_players[['FullName', 'Team', 'Position', 'CurrentPrice',
                        'TotalPoints', 'Minutes', 'PPM']]

show_all_risk = st.checkbox("Show all risk players", key="risk_all")
st.dataframe(
    risk_top.head(10) if not show_all_risk else risk_top,
    column_config={
        "CurrentPrice": st.column_config.NumberColumn(format="£%.1f"),
        "PPM": st.column_config.NumberColumn(format="%.1f"),
        "Minutes": st.column_config.ProgressColumn(
            format="%d",
            min_value=0,
            max_value=2610  # till week 29 (29*90)
        )
    },
    hide_index=True,  # <--- ADD THIS LINE
    height=400,
    use_container_width=True
)

# Visual 1: Interactive Bubble Plot with Plotly (Corrected)
st.header("Top Value Players: Points per Million(PPM) visualization")
st.write("""
Interactive bubble plot showing relationship between player price, total points, and minutes played.
- **Hover over bubbles** to see player details
- **Drag to zoom** | **Double-click to reset**
- Size of bubbles = minutes played
- Color intensity = Points per Million (PPM)
""")

# Add minutes filter
min_minutes = st.slider(
    "Select minimum minutes played:",
    min_value=0,
    max_value=2610,
    value=0,
    step=250,
    key='minutes_filter'
)

# Create filtered Plotly figure
plot_df = players_df[
    (players_df['Minutes'] >= min_minutes) &
    (players_df['Position'].notna()) &
    (players_df['Minutes'] > 0)
].copy()

fig1 = px.scatter(
    plot_df,
    x='CurrentPrice',
    y='TotalPoints',
    size='Minutes',
    color='PPM',
    hover_name='FullName',
    hover_data={
        'Team': True,
        'Position': True,
        'CurrentPrice': ':.1f',
        'TotalPoints': True,
        'Minutes': True,
        'PPM': ':.1f'
    },
    color_continuous_scale='Viridis',
    size_max=40,
    labels={
        'CurrentPrice': 'Price (£ million)',
        'TotalPoints': 'Total Points',
        'PPM': 'Points per Million'
    }
)

# Update layout (keep existing layout code)
fig1.update_layout(
    title=f'FPL Player Value: Points per Million (PPM) - Minimum {min_minutes} Minutes',
    plot_bgcolor='white',
    hovermode='closest',
    coloraxis_colorbar={
        'title': 'PPM',
        'title_side': 'right'
    }
)

# Remove the manual legend creation
fig1.update_layout(showlegend=False)  # Remove the old legend

# Set consistent axis ranges
max_price = plot_df['CurrentPrice'].max() * 1.1
max_points = plot_df['TotalPoints'].max() * 1.1
fig1.update_xaxes(range=[0, max_price])
fig1.update_yaxes(range=[0, max_points])

# Display in Streamlit
st.plotly_chart(fig1, use_container_width=True)

# Visual 2 Average PPM by Position
st.header("Positional Value Analysis")
st.write("Which position has the best value for money?")

fig2, ax = plt.subplots(figsize=(9, 3.5))  # Explicitly create figure and axis
positions_order = ['GK', 'DEF', 'MID', 'FWD']
ppm_by_position = players_df.groupby('Position')['PPM'].mean().reindex(positions_order)

# Create normalized color scale
norm = plt.Normalize(ppm_by_position.min() - 0.5, ppm_by_position.max())  # Adjust the min value
colors = plt.cm.Blues(norm(ppm_by_position))

# Plot on the explicit axis
bars = ppm_by_position.plot(kind='bar', color=colors, ax=ax)
plt.title('Average PPM by Position')
plt.xlabel('Position')
plt.ylabel('Average PPM')

# Add value labels
for idx, value in enumerate(ppm_by_position):
    ax.text(idx, value + 0.02, f'{value:.2f}', 
            ha='center', va='bottom', color='darkblue', fontsize=8)  # Reduced font size

# Add colorbar using the explicit axis
sm = plt.cm.ScalarMappable(norm=norm, cmap='Blues')
sm.set_array([])
plt.colorbar(sm, ax=ax, label='PPM Intensity Scale')  # Add ax=ax here

st.pyplot(fig2)
plt.close()


# Visual 3: Position-Specific Performance Factors Impact
st.header("Performance Factors Impact  — By Position")
st.write("Most common key performance factors for each position")

# Define factors for each position
position_factors = {
    'GK': ['CleanSheets', 'Saves', 'PenaltiesSaved', 'GoalsConceded', 'OwnGoals', 'YellowCards', 'RedCards'],
    'DEF': ['GoalsScored', 'Assists', 'CleanSheets', 'GoalsConceded', 'OwnGoals', 'YellowCards', 'RedCards'],
    'MID': ['GoalsScored', 'Assists', 'PenaltiesMissed', 'OwnGoals', 'YellowCards', 'RedCards'],
    'FWD': ['GoalsScored', 'Assists', 'PenaltiesMissed', 'OwnGoals', 'YellowCards', 'RedCards']
}

# Create 2x2 subplot grid
fig3, axs = plt.subplots(2, 2, figsize=(16, 12))
positions = ['GK', 'DEF', 'MID', 'FWD']
titles = ['Goalkeepers (GK)', 'Defenders (DEF)', 'Midfielders (MID)', 'Forwards (FWD)']

# Mapping for factor names with signs
factor_names_with_signs = {
    'CleanSheets': 'Clean Sheets (+)',
    'Saves': 'Saves (+)',
    'PenaltiesSaved': 'Penalties Saved (+)',
    'GoalsConceded': 'Goals Conceded (-)',
    'OwnGoals': 'Own Goals (-)',
    'YellowCards': 'Yellow Cards (-)',
    'RedCards': 'Red Cards (-)',
    'GoalsScored': 'Goals Scored (+)',
    'Assists': 'Assists (+)',
    'PenaltiesMissed': 'Penalties Missed (-)'
}

for i, pos in enumerate(positions):
    ax = axs[i//2, i%2]  # 2-row grid access
    pos_df = players_df[players_df['Position'] == pos]
    
    # Calculate average for each factor
    factors = position_factors[pos]
    avg_factors = pos_df[factors].mean().sort_values(ascending=False)
    
    # Create color mapping
    colors = plt.cm.Greens(np.linspace(0.5, 0.9, len(avg_factors)))  # Color by average value
    
    # Plot bars
    bars = ax.bar([factor_names_with_signs[factor] for factor in avg_factors.index], avg_factors.values, color=colors)
    
    # Formatting
    ax.set_title(titles[i], fontweight='bold')
    ax.set_ylabel('Average Value')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels over each bar
    for idx, value in enumerate(avg_factors):
        ax.text(
            idx, value, f'{value:.2f}',
            ha='center', va='bottom', color='dimgray', fontsize=8
        )

plt.tight_layout(pad=3.0)
st.pyplot(fig3)
plt.close()

# Visual 4: Position-Specific Performance Factors Impact
st.header("Performance Factors Impact on Total Points — By Position")
st.write("Correlation of key performance factors with Total Points for each position")

# Define factors for each position
position_factors = {
    'GK': ['CleanSheets', 'Saves', 'PenaltiesSaved', 'GoalsConceded', 'OwnGoals', 'YellowCards', 'RedCards'],
    'DEF': ['GoalsScored', 'Assists', 'CleanSheets', 'GoalsConceded', 'OwnGoals', 'YellowCards', 'RedCards'],
    'MID': ['GoalsScored', 'Assists', 'PenaltiesMissed', 'OwnGoals', 'YellowCards', 'RedCards'],
    'FWD': ['GoalsScored', 'Assists', 'PenaltiesMissed', 'OwnGoals', 'YellowCards', 'RedCards']
}

# Create 2x2 subplot grid
fig3, axs = plt.subplots(2, 2, figsize=(16, 12))
positions = ['GK', 'DEF', 'MID', 'FWD']
titles = ['Goalkeepers (GK)', 'Defenders (DEF)', 'Midfielders (MID)', 'Forwards (FWD)']

for i, pos in enumerate(positions):
    ax = axs[i//2, i%2]  # 2-row grid access
    pos_df = players_df[players_df['Position'] == pos]
    
    # Calculate correlations
    factors = position_factors[pos]
    corr = pos_df[factors + ['TotalPoints']].corr()['TotalPoints'].drop('TotalPoints')
    
    # Sort by absolute impact
    corr_sorted = corr.sort_values(key=lambda x: abs(x), ascending=False)
    
    # Create color mapping
    colors = plt.cm.YlOrRd(np.abs(corr_sorted.values))  # Color by impact strength
    
    # Plot bars with absolute values to ensure all are positive
    bars = ax.bar([factor_names_with_signs[factor] for factor in corr_sorted.index], np.abs(corr_sorted.values), color=colors)
    
    # Formatting
    ax.set_title(titles[i], fontweight='bold')
    ax.set_ylabel('Correlation with Total Points')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ensure consistent y-axis ticks for all positions
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    
    # Add value labels
    for idx, value in enumerate(np.abs(corr_sorted)):
        ax.text(
            idx, value + 0.02, f'{value:.2f}',  # Adjusted y_offset for positive values
            ha='center', va='bottom', color='dimgray', fontsize=8
        )

plt.tight_layout(pad=3.0)
st.pyplot(fig3)
plt.close()

