# Football Match Report Generator

A reusable python engine for generating data-driven football match
visualisations using Sofascore data.

## Features
- Scrapes match data from Sofascore
- Average player position figrues with attacking zone usage
- Match momentum graph
- Cumulative xG graph
- Shot maps for each team

## Usage 
The functions in the match report generator are designed to be imported
to a Jupyter Notebook and used with a Sofascore match ID. 

1. Import the engine
from match_report_utils import *

2. Provide a Sofascore match ID
match_id = 1234567
The match ID can be found for any game in the Sofascore URL

3. Generate Visuals
Individual functions can then be called to generate visuals
to summaries the match.

- scrape_data(match_id)
- plot_average_positions(team, team_colour, home_team, interactive = False)
- plot_momentum(home_colour, away_colour)
- plot_xg(home_team, home_colour, away_team, away_colour)
- plot_shot_maps(home_team, away_team)

The generated figures are saved as PNG files for use in a match report,
with the exception of the average positions plotting function which has
the option to save the figures as interactive html using plotly.

The engine is designed to be modular, allowing individual visualisations 
to be generated independently depending on the requirements of the report.