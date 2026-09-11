# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ScraperFC as sfc
from pathlib import Path 
import pickle
from mplsoccer import Pitch, VerticalPitch
from adjustText import adjust_text
import ast
import plotly.graph_objects as go

# Initialise sofascore
sofa = sfc.Sofascore()


# ----------------------------------------------
# DATA SCRAPING FUNCTION
# ----------------------------------------------

def scrape_data(match_id):

    # Find the folder containing the current notebook
    match_folder = Path.cwd()
    data_folder = match_folder / "data"

    # Create data folder if it doesn't already exist
    data_folder.mkdir(exist_ok=True)

    # Scrape the data
    player_ids = sofa.get_match_player_ids(match_id)
    heatmaps = sofa.scrape_heatmaps(match_id)
    momentum = sofa.scrape_match_momentum(match_id)
    shots = sofa.scrape_match_shots(match_id)
    avg_positions = sofa.scrape_player_average_positions(match_id)
    player_stats = sofa.scrape_player_match_stats(match_id)
    team_stats = sofa.scrape_team_match_stats(match_id)

    # Save the DataFrames as csv files
    momentum.to_csv(data_folder / "momentum.csv", index=False)
    shots.to_csv(data_folder / "shots.csv", index=False)
    avg_positions.to_csv(data_folder / "average_positions.csv", index=False)
    player_stats.to_csv(data_folder / "player_stats.csv", index=False)
    team_stats.to_csv(data_folder / "team_stats.csv", index=False)
    
        
    # Save dictionaries as pickle files
    with open(data_folder / "player_ids.pkl", "wb") as f:
        pickle.dump(player_ids, f)

    with open(data_folder / "heatmaps.pkl", "wb") as f:
        pickle.dump(heatmaps, f)


# --------------------------------------------------
# HELPER FUNCTIONS FOR AVERAGE POSITION PLOTS
# --------------------------------------------------

# Helper function for OPPDA
def get_team_stat(team_stats, stat_name):
    row = team_stats[team_stats["name"] == stat_name].iloc[0]
            
    return {
    "home": float(row["home"]),
    "away": float(row["away"])}


# Helper function to define pressing style
def get_pressing_style(oppda):
    if oppda <= 9:
        return "HIGH PRESS"
    elif oppda <= 12:
        return "MID BLOCK"
    else:
        return "LOW BLOCK"


# Helper function for drawing a plotly pitch
def add_plotly_pitch(fig):

    # Pitch colours
    pitch_colour = "#404040"
    line_colour = "#CCCCCC"

    # Pitch dimensions
    pitch_length = 105
    pitch_width = 68
    
    # Pitch outline
    fig.add_shape(
        type = "rect",
        x0 = 0,
        y0 = 0,
        x1 = pitch_length,
        y1 = pitch_width,
        line = dict(color = line_colour,
                    width = 1),
        fillcolor = pitch_colour,
        layer = "below")

    # Halfway line
    fig.add_shape(
        type = "line",
        x0 = 52.5,
        y0 = 0,
        x1 = 52.5,
        y1 = pitch_width,
        line = dict(color = line_colour,
                   width = 1),
        layer = "below")

    # Centre Circle
    centre_x = 52.5
    centre_y = 34
    centre_radius = 9.15
    
    fig.add_shape(
        type = "circle",
        x0 = centre_x - centre_radius,
        y0 = centre_y - centre_radius,
        x1 = centre_x + centre_radius,
        y1 = centre_y + centre_radius,
        line = dict(color = line_colour,
                   width = 1),
        layer = "below")

    # Penalty areas
    penalty_length = 16.5
    penalty_width = 40.32

    penalty_y0 = (pitch_width - penalty_width) / 2
    penalty_y1 = (pitch_width + penalty_width) / 2

    # Left
    fig.add_shape(
        type = "rect",
        x0 = 0,
        y0 = penalty_y0,
        x1 = penalty_length,
        y1 = penalty_y1,
        line = dict(color = line_colour,
                    width=1),
        fillcolor = "rgba(0,0,0,0)",
        layer = "below")

    # Right
    fig.add_shape(
        type = "rect",
        x0 = pitch_length - penalty_length,
        y0 = penalty_y0,
        x1 = pitch_length,
        y1 = penalty_y1,
        line = dict(color = line_colour,
                    width=1),
        fillcolor = "rgba(0,0,0,0)",
        layer = "below"
    )

    # Six-yard boxes
    six_yard_length = 5.5
    six_yard_width = 18.32

    six_y0 = (pitch_width - six_yard_width) / 2
    six_y1 = (pitch_width + six_yard_width) / 2

    # Left
    fig.add_shape(
        type = "rect",
        x0 = 0,
        y0 = six_y0,
        x1 = six_yard_length,
        y1 = six_y1,
        line = dict(color=line_colour,
                    width=1),
        fillcolor = "rgba(0,0,0,0)",
        layer = "below")

    # Right
    fig.add_shape(
        type = "rect",
        x0 = pitch_length - six_yard_length,
        y0 = six_y0,
        x1 = pitch_length,
        y1 = six_y1,
        line = dict(color = line_colour,
                    width = 1),
        fillcolor = "rgba(0,0,0,0)",
        layer = "below")

    
    # Penalty spots
    penalty_spot = 11
    
    fig.add_trace(
        go.Scatter(
            x = [penalty_spot, pitch_length - penalty_spot],
            y = [centre_y, centre_y],
            mode = "markers",
            marker = dict(size = 5,
                          color=line_colour),
            hoverinfo = "skip",
            showlegend = False))


    # Penalty arcs
    penalty_arc_radius = 9.15

    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    
    # Left D
    left_arc_x = penalty_spot + penalty_arc_radius * np.cos(theta)
    left_arc_y = centre_y + penalty_arc_radius * np.sin(theta)
    
    left_mask = left_arc_x >= penalty_length
    
    fig.add_trace(
        go.Scatter(
            x=left_arc_x[left_mask],
            y=left_arc_y[left_mask],
            mode="lines",
            line=dict(
                color=line_colour,
                width=1
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )
    
    # Right D
    right_arc_x = pitch_length - penalty_spot - penalty_arc_radius * np.cos(theta)
    right_arc_y = centre_y + penalty_arc_radius * np.sin(theta)
    
    right_mask = right_arc_x <= pitch_length - penalty_length
    
    fig.add_trace(
        go.Scatter(
            x=right_arc_x[right_mask],
            y=right_arc_y[right_mask],
            mode="lines",
            line=dict(
                color=line_colour,
                width=1
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Center circle
    fig.add_trace(go.Scatter(x = [52.5],
                             y = [34],
                             mode = "markers",
                             marker = dict(size = 5,
                                           color = line_colour),
                             hoverinfo = "skip",
                             showlegend = False))
        

    # Plot dimensions
    fig.update_layout(width = 560,
                      height = 363,

                      margin = dict(l = 2,
                                    r = 2,
                                    t = 20,
                                    b = 10),

                      plot_bgcolor = "#404040",
                      paper_bgcolor = "#404040",
                                  
                      xaxis = dict(range=[0, 105],
                                   showgrid = False,
                                   showticklabels = False,
                                   zeroline = False),

                      yaxis = dict(range = [0, 68],
                                 showgrid = False,
                                 showticklabels = False,
                                 zeroline = False,
                                 scaleanchor = "x",
                                 scaleratio = 1)
                      )

# --------------------------------------------------
# AVERAGE POSITIONS FUNCTION
# --------------------------------------------------
def plot_average_positions(
        team,
        team_colour,
        home_team,
        interactive = False):
    
    # File locations
    data_folder = Path("data")

    # Figures folder for non interactive plots
    figures_folder = Path("figures")
    figures_folder.mkdir(exist_ok = True)

    # Html folder for interactive plots
    html_folder = Path("html")
    html_folder.mkdir(exist_ok=True)
    
    # Load data
    avg_positions = pd.read_csv(data_folder / "average_positions.csv")
    team_stats = pd.read_csv(data_folder / "team_stats.csv")
    player_stats = pd.read_csv(data_folder / "player_stats.csv")
    
    with open(data_folder / "heatmaps.pkl", "rb") as f:
        heatmaps = pickle.load(f)


    # Find the starting XI
    avg_positions = avg_positions[["shortName", "jerseyNumber", 
                                   "id", "averageX", "averageY","team"]]
    player_stats = player_stats[["id","substitute", "minutesPlayed", "touches",
                                "topSpeed", "kilometersCovered"]]

    avg_positions = avg_positions.merge(player_stats,
                                       on = "id",
                                       how = "left")
    
    starting_xis = avg_positions[avg_positions["substitute"] == False].copy()
    team_xi = starting_xis[starting_xis["team"] == team].copy()    
    
    
    
    # --------- OPPDA ----------
    passes = get_team_stat(team_stats, "Passes")
    tackles = get_team_stat(team_stats, "Total tackles")
    interceptions = get_team_stat(team_stats, "Interceptions")
    fouls = get_team_stat(team_stats, "Fouls")

    if team == home_team:
        OPPDA = (passes["away"] / (tackles["home"] + interceptions["home"] + fouls["home"]))
    else:
        OPPDA = (passes["home"] / (tackles["away"] + interceptions["away"] + fouls["away"]))

    # Classify pressing style
    pressing_style = get_pressing_style(OPPDA)
    
    # Define default font
    plt.rcParams["font.family"] = "DIN Alternate"

    
    # --------- Final Third Zones ------------
    # Get all players belonging to the team
    team_players = avg_positions[avg_positions["team"] == team][["id", "shortName"]]

    # Organise heatmaps by player ID
    heatmaps_by_id = {
        player_data["id"]: player_data["heatmap"]
        for player_data in heatmaps.values()}

    # Get all heatmap actions for the team
    team_actions = []

    for player_id in team_players["id"]:
    
        if player_id in heatmaps_by_id:
            team_actions.extend(heatmaps_by_id[player_id])
    
    # Get actions in the final third
    final_third_actions = [
    (x, y) for x, y in team_actions
    if x >= 66.67]

    # Count final-third actions by zone
    left_actions = 0
    central_actions = 0
    right_actions = 0
    
    for x, y in final_third_actions:
    
        if y >= 66.67:
            left_actions += 1
    
        elif y >= 33.33:
            central_actions += 1
    
        else:
            right_actions += 1

    total_final_third = len(final_third_actions)

    left_pct = left_actions / total_final_third * 100
    central_pct = central_actions / total_final_third * 100
    right_pct = right_actions / total_final_third * 100



    
    

    # ----------------- Split between interactive and non-interactive-----------------

    # V2 - Interactive
    if interactive:

        # Convert player positions from OPTA 0 x 100 coordinates
        # to display 105 x 68 coordinates
        team_xi = starting_xis[starting_xis["team"] == team].copy()
        
        team_xi["plot_x"] = team_xi["averageX"] * 105 / 100
        team_xi["plot_y"] = team_xi["averageY"] * 68 / 100

        # Add custom data to show on plot
        customdata = team_xi[["id", "shortName", "minutesPlayed",
                             "touches", "topSpeed", "kilometersCovered"]].to_numpy()

        # Establish figure
        fig = go.Figure()

        add_plotly_pitch(fig)

        # First trace for player positions and shirt numbers and hover info
        fig.add_trace(go.Scatter(x = team_xi["plot_x"],
                                 y = team_xi["plot_y"],
                                 mode = "markers+text",

                                 # Shirt numbers inside the player points
                                 text = team_xi["jerseyNumber"],
                                 textposition = "middle center",
                                 textfont = dict(size = 16,
                                                color = "white",
                                                family = "DIN Alternate"),
                                             
                                 marker = dict(size = 30,
                                              color = team_colour),

                                 customdata = customdata,
                                             
                                 hovertemplate = (
                                            "<b>%{customdata[1]}</b><br>"
                                            "Minutes: %{customdata[2]}<br>"
                                            "Touches: %{customdata[3]}<br>"
                                            "Top speed: %{customdata[4]:.1f} km/h<br>"
                                            "Distance: %{customdata[5]:.1f} km"
                                            "<extra></extra>"),

                                 hoverlabel = dict(
                                     bgcolor = team_colour,
                                     bordercolor = "white",
                                     font = dict(color = "white",
                                                size = 12,
                                                family = "DIN Alternate")),
                                 
                                 showlegend = False))

        # Second trace for player names
        fig.add_trace(go.Scatter(x = team_xi["plot_x"],
                                 y = team_xi["plot_y"] - 5,
                                 mode = "text",
                                
                                 text = team_xi["shortName"],                                
                                 textfont = dict(size = 12,
                                                color = "white",
                                                family = "DIN Alternate"),
                                 
                                 hoverinfo = "skip",
                                 showlegend = False))

        # Recalculate the final third zones for new coordinate system
        final_third_x = 66.67 * 105 / 100

        left_y_min = 66.67 * 68 / 100
        left_y_max = 68
        
        central_y_min = 33.33 * 68 / 100
        central_y_max = 66.67 * 68 / 100
        
        right_y_min = 0
        right_y_max = 33.33 * 68 / 100

        # Scale alpha relative to the three zones
        min_pct = min(left_pct, central_pct, right_pct)
        max_pct = max(left_pct, central_pct, right_pct)
        
        min_alpha = 0.25
        max_alpha = 0.75
        
        def scale_alpha(pct):
            if max_pct == min_pct:
                return (min_alpha + max_alpha) / 2
        
            return min_alpha + (
                (pct - min_pct) / (max_pct - min_pct)
            ) * (max_alpha - min_alpha)
        
        left_alpha = scale_alpha(left_pct)
        central_alpha = scale_alpha(central_pct)
        right_alpha = scale_alpha(right_pct)
        
        # Add the attacking zones to the pitch
        zone_shape_indices = []
        
        fig.add_shape(type = "rect",
                      x0 = final_third_x,
                      x1 = 105,
                      y0 = left_y_min,
                      y1 = left_y_max,
                      fillcolor = team_colour,
                      opacity = left_alpha,
                      line = dict(color=team_colour,
                                width=1),
                      layer = "below")
        zone_shape_indices.append(len(fig.layout.shapes) - 1)

        fig.add_shape(type = "rect",
                              x0 = final_third_x,
                              x1 = 105,
                              y0 = central_y_min,
                              y1 = central_y_max,
                              fillcolor = team_colour,
                              opacity = central_alpha,
                              line = dict(color = team_colour,
                                        width = 1),
                              layer = "below")
        zone_shape_indices.append(len(fig.layout.shapes) - 1)
        
        fig.add_shape(type = "rect",
                              x0 = final_third_x,
                              x1 = 105,
                              y0 = right_y_min,
                              y1 = right_y_max,
                              fillcolor = team_colour,
                              opacity = right_alpha,
                              line = dict(color = team_colour,
                                        width = 1),
                              layer = "below")
        zone_shape_indices.append(len(fig.layout.shapes) - 1)

        # Annotate the attacking zone percentages
        zone_annotation_indices = []
        
        fig.add_annotation(x = 98,
                           y = (left_y_min + left_y_max) / 2,
                           text = f"{left_pct:.1f}%",
                           showarrow = False,
                           font = dict(size = 16,
                                       color = "#CCCCCC",
                                       family = "DIN Alternate"))
        zone_annotation_indices.append(len(fig.layout.annotations) - 1)

        fig.add_annotation(x = 98,
                           y = (central_y_min + central_y_max) / 2,
                           text = f"{central_pct:.1f}%",
                           showarrow = False,
                           font = dict(size = 16,
                                       color = "#CCCCCC",
                                       family = "DIN Alternate"))
        zone_annotation_indices.append(len(fig.layout.annotations) - 1)


        fig.add_annotation(x = 98,
                           y = (right_y_min + right_y_max) / 2,
                           text = f"{right_pct:.1f}%",
                           showarrow = False,
                           font = dict(size = 16,
                                       color = "#CCCCCC",
                                       family = "DIN Alternate"))
        zone_annotation_indices.append(len(fig.layout.annotations) - 1)

        show_zones = {}

        for index in zone_shape_indices:
            show_zones[f"shapes[{index}].visible"] = True
        
        for index in zone_annotation_indices:
            show_zones[f"annotations[{index}].visible"] = True
        
        
        hide_zones = {}
        
        for index in zone_shape_indices:
            hide_zones[f"shapes[{index}].visible"] = False
        
        for index in zone_annotation_indices:
            hide_zones[f"annotations[{index}].visible"] = False

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x = 0.96,
                    y = 0.06,
                    xanchor = "right",
                    yanchor = "middle",
                    showactive = True,
                    
                    font = dict(size = 12,
                               family = "DIN Alternate",
                               color = "#404040"),
                    pad = dict(
                        l = 1,
                        r = 1,
                        t = 1,
                        b = 1),

                    borderwidth = 1, 
                    
                    buttons=[
                        dict(
                            label="HIDE",
                            method="relayout",
                            args = [show_zones],
                            args2 = [hide_zones])])],

                    title=dict(text=f"{team} Starting XI Average Positions",
                                    x=0.04,
                                    xanchor="left",
                                    y=0.98,
                                    yanchor="middle",
                                    font=dict(
                                    size=16,
                                    color="white",
                                    family = "DIN Alternate")))

        fig.add_annotation(x = 93,
                           y = 2,
                           text = "ATTACKING ZONES",
                           showarrow = False,
                           xanchor = "right",
                           yanchor = "middle",
                           font = dict(size = 12,
                                       color = "#CCCCCC",
                                       family = "DIN Alternate"))
        
        # Annotate the OPPDA
        fig.add_annotation(x = 2,
                           y = 2,
                           text = f"{OPPDA:.1f} OPPDA | {pressing_style}",
                           showarrow = False,
                           xanchor = "left",
                           yanchor = "middle",
                           font = dict(size = 12,
                                       color = "#CCCCCC",
                                       family = "DIN Alternate"))

        # Add heatmaps which reveal once hovering
        heatmap_trace_indices = {}

        for _, player in team_xi.iterrows():
        
            player_id = player["id"]
        
            if player_id not in heatmaps_by_id:
                continue
        
            player_touches = heatmaps_by_id[player_id]
        
            xs = [x for x, y in player_touches]
            ys = [y for x, y in player_touches]
        
            plot_xs = [x * 105 / 100 for x in xs]
            plot_ys = [y * 68 / 100 for y in ys]
        
            fig.add_trace(
                go.Histogram2dContour(
                    x=plot_xs,
                    y=plot_ys,

                    nbinsx=10,
                    nbinsy=10,
        
                    colorscale=[
                        [0.00, "rgba(0,0,0,0)"],
                        [0.20, "rgba(255,255,255,0.05)"],
                        [0.45, team_colour],
                        [0.70, team_colour],
                        [1.00, team_colour]
                    ],
        
                    showscale=False,
        
                    ncontours=50,
        
                    contours=dict(
                        coloring="fill"
                    ),
        
                    opacity=0.65,
        
                    line=dict(
                        width=0
                    ),
        
                    hoverinfo="skip",
        
                    showlegend=False,
        
                    visible=False
                )
            )
        
            heatmap_trace_indices[str(player_id)] = len(fig.data) - 1
    

        post_script = f"""
        var plot = document.getElementById('{{plot_id}}');
        
        var heatmaps = {heatmap_trace_indices};
        var heatmapIndices = Object.values(heatmaps);
        
        plot.on('plotly_hover', function(data) {{
        
            var point = data.points[0];
        
            if (!point.customdata) {{
                return;
            }}
        
            var playerId = String(point.customdata[0]);
        
            // Hide all heatmaps
            Plotly.restyle(
                plot,
                {{visible: false}},
                heatmapIndices
            );
        
            // Show hovered player's heatmap
            if (heatmaps[playerId] !== undefined) {{
        
                Plotly.restyle(
                    plot,
                    {{visible: true}},
                    [heatmaps[playerId]]
                );
        
            }}
        
        }});
        
        plot.on('plotly_unhover', function(data) {{
        
            Plotly.restyle(
                plot,
                {{visible: false}},
                heatmapIndices
            );
        
        }});
        """
        fig.write_html(
            html_folder / f"{team}_average_positions.html",
            auto_open=False,
            post_script=post_script,
            config = {"responsive" : True}
        )

        fig.show()
            
    # V1 - Non-interactive ------------------------------------------
    else:
        # Establish pitch and figure
        pitch = Pitch(pitch_type = "opta",
                      pitch_color = "#404040",
                      line_color = "#CCCCCC",
                      linewidth = 0.75)
    
        fig, ax = pitch.draw(figsize=(16, 9))
        
        # Plot home average positions
        pitch.scatter(team_xi["averageX"],
                      team_xi["averageY"],
                      ax = ax,
                      s = 1000,
                      color = team_colour,
                      zorder=3)
    
        # Annotate player names and shirt numbers
        for _, player in team_xi.iterrows():
            
            # Player Names
            pitch.annotate(
                player["shortName"],
                xy = (player["averageX"], player["averageY"]),
                xytext = (0,-20),
                textcoords = "offset points",
                ha = "center",
                va = "top",
                fontsize = 16,
                color = "white",
                ax = ax)
    
            # Shirt Numbers
            pitch.annotate(
                str(player["jerseyNumber"]),
                xy = (player["averageX"], player["averageY"]),
                ha = "center",
                va = "center",
                fontsize = 18,
                fontweight = "bold",
                color = "white",
                ax = ax)
    
        # Annotate the OPPDA
        ax.text(2,2,
                f"{OPPDA:.1f} OPPDA | {pressing_style}",
                fontsize = 16,
                color = "#CCCCCC",
                ha = "left",
                va = "bottom")
        
        # Attacking Zones
        final_third_x = 66.67
    
        # Get the actual pitch limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Scale alpha relative to the three zones
        min_pct = min(left_pct, central_pct, right_pct)
        max_pct = max(left_pct, central_pct, right_pct)
        
        min_alpha = 0.25
        max_alpha = 0.75
        
        def scale_alpha(pct):
            if max_pct == min_pct:
                return (min_alpha + max_alpha) / 2
        
            return min_alpha + (
                (pct - min_pct) / (max_pct - min_pct)
            ) * (max_alpha - min_alpha)
        
        left_alpha = scale_alpha(left_pct)
        central_alpha = scale_alpha(central_pct)
        right_alpha = scale_alpha(right_pct)
        
        
        # Convert SofaScore y-coordinates to pitch coordinates
        left_y_min = 66.67
        left_y_max = 100
        
        central_y_min = 33.33
        central_y_max = 66.67
        
        right_y_min = 0
        right_y_max = 33.33
        
        
        # Left zone
        ax.fill_between(
            [final_third_x, 100],
            left_y_min,
            left_y_max,
            color = team_colour,
            alpha = left_alpha,
            zorder = 0)
        ax.text(95, 83.33,
                f"{left_pct:.1f}%",
                fontsize = 20,
                fontweight = "bold",
                color = "#CCCCCC",
                ha = "right",
                va = "center",
                zorder = 5)
        
        # Central zone
        ax.fill_between(
            [final_third_x, 100],
            central_y_min,
            central_y_max,
            color = team_colour,
            alpha = central_alpha,
            zorder = 0)
        ax.text(95, 50,
                f"{central_pct:.1f}%",
                fontsize = 20,
                fontweight = "bold",
                color = "#CCCCCC",
                ha = "right",
                va = "center",
                zorder = 5)
        
        # Right zone
        ax.fill_between(
            [final_third_x, 100],
            right_y_min,
            right_y_max,
            color = team_colour,
            alpha = right_alpha,
            zorder = 0)
        ax.text(95, 17.67,
                f"{right_pct:.1f}%",
                fontsize = 20,
                fontweight = "bold",
                color = "#CCCCCC",
                ha = "right",
                va = "center",
                zorder = 5)
    
        ax.text(98,2,
               "ATTACKING ZONES",
                fontsize = 16,
                color = "#CCCCCC",
                ha = "right",
                va = "bottom")
        
        # Set title
        ax.set_title(f"{team} Starting XI Average Positions",
                          fontsize = 18,
                          color = "#2E2E2E",
                          pad = 10,
                          loc = "left")
    
        # Save fig to figures folder
        fig.savefig(
            figures_folder / f"{team}_average_positions.png",
            format = "png",
            dpi = 150,
            bbox_inches = "tight")
    
        plt.show()


# --------------------------------------------------------------------------



# --------------------------------------------------
# MOMENTUM GRAPH
# --------------------------------------------------
def plot_momentum(home_colour, away_colour):

    # File locations
    data_folder = Path("data")
    figures_folder = Path("figures")

    # Load the data
    momentum_data = pd.read_csv(data_folder / "momentum.csv")

    # Define default font
    plt.rcParams["font.family"] = "DIN Alternate"

    # Create a figire and axes
    fig, ax = plt.subplots(1,1, figsize = (20,6))

    # Set background colour
    ax.set_facecolor("#404040")

    # Copy the momentum data
    momentum_plot = momentum_data.copy()
    
    # Find where momentum changes sign
    crossings = momentum_plot["value"].shift(1) * momentum_plot["value"] < 0
    
    # Add zero-crossing points
    new_rows = []
    
    for i in momentum_plot.index[crossings]:
        
        x1 = momentum_plot.loc[i - 1, "minute"]
        x2 = momentum_plot.loc[i, "minute"]
        
        # Momentum crosses exactly at zero
        crossing_minute = x1 + ((0 - momentum_plot.loc[i - 1, "value"]) /
                                (momentum_plot.loc[i, "value"] - momentum_plot.loc[i - 1, "value"])) * (x2 - x1)
        
        new_rows.append({"minute": crossing_minute,
                         "value": 0})
    
    # Add the crossing points and sort by minute
    if new_rows:
        momentum_plot = pd.concat([momentum_plot, pd.DataFrame(new_rows)],
                                  ignore_index=True)
    
        momentum_plot = momentum_plot.sort_values("minute").reset_index(drop=True)
    
    
    # Create positive and negative values
    home_momentum = momentum_plot["value"].where(momentum_plot["value"] >= 0)
    
    away_momentum = momentum_plot["value"].where(momentum_plot["value"] <= 0)

    ax.fill_between(momentum_plot["minute"],
                   home_momentum,
                   0,
                   color = home_colour,
                   alpha = 0.9)

    # Fill the momentum for the away team
    ax.fill_between(momentum_plot["minute"],
                   away_momentum,
                   0,
                   color = away_colour,
                   alpha = 0.9)

    # Horizontal line at y = 0 to split home and away momentum
    ax.axhline(0,
            color = "#CCCCCC",
            linewidth = 1)

    # Vertical Line to split halves
    ax.axvline(45,
            color = "#CCCCCC",
            linewidth = 1,
            linestyle = "--")

    # Change axes scales
    ax.set_xticks(range(0,91,5))
    ax.set_yticks([])

    # Set title
    ax.set_title("Match Momentum",
                fontsize = 18,
                color = "#2E2E2E",
                pad = 10,
                loc = "left")
    
    # Set x axis label
    ax.set_xlabel("Minutes",
                 fontsize = 14,
                 color = "#2E2E2E",
                 loc = "center")

    # Labels for each half
    ax.text(22.5,momentum_data["value"].min(),
            "FIRST HALF",
            fontsize = 14,
            color = "#CCCCCC",
            ha = "center",
            va = "bottom")

    ax.text(67.5,momentum_data["value"].min(),
            "SECOND HALF",
            fontsize = 14,
            color = "#CCCCCC",
            ha = "left",
            va = "bottom")
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    # Save the figure
    fig.savefig(figures_folder / "momentum_graph.png",
             format = "png",
             dpi = 150,
             bbox_inches = "tight")

    plt.show()


# ---------------------------------------------------------



# --------------------------------------------------
# CUMULATIVE XG GRAPH
# --------------------------------------------------
def plot_xg(home_team, home_colour, 
            away_team, away_colour):

    # File locations
    data_folder = Path("data")
    figures_folder = Path("figures")

    # Load the shot data 
    shot_data = pd.read_csv(data_folder / "shots.csv")
    
    # Define default font
    plt.rcParams["font.family"] = "DIN Alternate"
    
    # Sort the shots chronologically
    shot_data = shot_data.sort_values("time")
    
    # Split home and away shots
    home_shots = shot_data[shot_data["isHome"] == True].copy()
    away_shots = shot_data[shot_data["isHome"] == False].copy()

    # Cumalative xG
    home_shots["total_xG"] = home_shots["xg"].cumsum()
    away_shots["total_xG"] = away_shots["xg"].cumsum()

    # Combine home and away shots back together now with cumalative xG
    shot_data = pd.concat([home_shots, away_shots]).sort_values("time")
    
    # Extract the goals
    goals = shot_data[shot_data["shotType"] == "goal"].copy()
    
    # Extract goalscorer names
    goals["playerName"] = goals["player"].apply(
        lambda x: ast.literal_eval(x)["shortName"])
    
    # Add a zero point at kick-off and fulltime
    home_x = [0] + home_shots["time"].tolist() + [90]
    home_y = [0] + home_shots["total_xG"].tolist() + [home_shots["total_xG"].iloc[-1]]
    
    away_x = [0] + away_shots["time"].tolist() + [90]
    away_y = [0] + away_shots["total_xG"].tolist() + [away_shots["total_xG"].iloc[-1]]

    # Create a figure
    fig, ax = plt.subplots(1,1,figsize = (16,9))

    # Set background colour
    ax.set_facecolor("#404040")

    # Plot cumalative xG
    ax.step(home_x,
           home_y,
           where = "post",
           color = home_colour,
           linewidth = 2.5)
    ax.step(away_x,
           away_y,
           where = "post",
           color = away_colour,
           linewidth = 2.5)

    # Label Goalscorers
    for _, goal in goals.iterrows():

        # Circle at goal
        ax.scatter(goal["time"],
                  goal["total_xG"],
                  s = 100,
                  color = "#CCCCCC",
                  zorder = 5)
        
        # Annotate goalscorer name
        ax.annotate(
            f"{goal["playerName"]} - {goal["time"]}' ",
            xy = (goal["time"], goal["total_xG"]),
            xytext = (0,10),
            textcoords = "offset points",
            ha = "center",
            va = "bottom",
            fontsize = 14,
            color = "#CCCCCC",
            fontweight = "bold",
            zorder = 1)
    
    # X-ais 
    ax.set_xlim(0,95)
    ax.set_xticks(range(0,91,5))

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Labels
    ax.set_xlabel("Minutes",
                 fontsize = 14,
                 color = "#2E2E2E",
                 loc = "center")
    ax.set_ylabel("Cumalative xG",
                 fontsize = 14,
                 color = "#2E2E2E",
                 loc = "center")
    # Set the title
    ax.set_title(f"xG: {home_team} {home_shots["total_xG"].iloc[-1]:.2f} - "
                 f"{away_shots["total_xG"].iloc[-1]:.2f} {away_team}",
                 fontsize = 18,
                 color = "#2E2E2E",
                 pad = 10,
                 loc = "left")

    # Save the figure
    fig.savefig(figures_folder / "xg.png",
               format = "png",
               dpi = 150,
               bbox_inches = "tight")
    
    plt.show()


# ---------------------------------------------


# --------------------------------------------------
# SHOT MAP FUNCTION
# --------------------------------------------------
def plot_shot_maps(home_team, away_team):

    # Define the folders
    data_folder = Path("data")
    figures_folder = Path("figures")

    # Load the shot data
    shots = pd.read_csv(data_folder / "shots.csv")

    # Split x and y shot locations
    shots["playerCoordinates"] = shots["playerCoordinates"].apply(ast.literal_eval)

    shots["x"] = shots["playerCoordinates"].apply(lambda d:d["x"])
    shots["y"] = shots["playerCoordinates"].apply(lambda d:d["y"])

    # Transform the shots to the vertical, half pitch system
    shots["plot_x"] = 100 - shots["y"]
    shots["plot_y"] = 100 - shots["x"]

    # split shots by open play and set pieces
    open_play_situations = ["assisted", "fast-break", "regular"]

    shots["xg_type"] = np.where(shots["situation"].isin(open_play_situations),
                               "open_play",
                               "set_piece")
    
    # Split home and away shots
    home_shots = shots[shots["isHome"] == True].copy()
    away_shots = shots[shots["isHome"] == False].copy()

    # Calculate home and away open play and set piece xg
    home_open_xg = home_shots.loc[home_shots["xg_type"] == "open_play", "xg"].sum()
    home_set_piece_xg = home_shots["xg"].sum() - home_open_xg

    away_open_xg = away_shots.loc[away_shots["xg_type"] == "open_play", "xg"].sum()
    away_set_piece_xg = away_shots["xg"].sum() - away_open_xg
    
    # xG per shot calculation
    home_xg_per_shot = home_shots["xg"].mean()
    away_xg_per_shot = away_shots["xg"].mean()

    # Shooting quality: xGOT - xG
    home_qual = home_shots["xgot"].sum() - home_shots["xg"].sum()
    away_qual = away_shots["xgot"].sum() - away_shots["xg"].sum()

    # Define default font
    plt.rcParams["font.family"] = "DIN Alternate"

    # Define a colour map for the shot outcome
    shot_colours = {"goal" : "#3DE377",
                    "miss" : "#E33D56",
                    "save" : "#3DA9E3",
                    "block" : "#E3773D",
                    "post" : "#BC3DE3"}
    
    # --------- Create the figure -------------
    
    # Create the pitch
    pitch = VerticalPitch(pitch_type = "opta",
                  pitch_color = "#404040",
                  line_color = "#CCCCCC",
                  linewidth = 0.75,
                  half = True)
    
    fig, ax = pitch.draw(nrows = 1,
                         ncols = 2, 
                         figsize = (16,9))

   

    # Plot the home shots on the first subplot
    for _, shot in home_shots.iterrows():

        colour = shot_colours[shot["shotType"]]
        
        ax[0].scatter(shot["plot_x"],
                      shot["plot_y"],
                      s = np.sqrt(shot["xg"]) * 1000,
                      c = colour,
                      alpha = 0.9)
            

    for _, shot in away_shots.iterrows():

        colour = shot_colours[shot["shotType"]]
        
        ax[1].scatter(shot["plot_x"],
                      shot["plot_y"],
                      s = np.sqrt(shot["xg"]) * 1000,
                      c = colour,
                      alpha = 0.9)
    
    # Create dummy scatter points for the legened
    legend_elements = [
        Line2D([0], [0],
               marker = "o",
               color = "none",
               markerfacecolor = shot_colours["goal"],
               markeredgecolor = "white",
               markersize = 20,
               label = "Goal"),
        
        Line2D([0], [0],
               marker = "o",
               color = "none",
               markerfacecolor = shot_colours["save"],
               markeredgecolor = "white",
               markersize = 20,
               label = "Saved"),
        
        Line2D([0], [0],
               marker = "o",
               color = "none",
               markerfacecolor = shot_colours["block"],
               markeredgecolor = "white",
               markersize = 20,
               label = "Blocked"),
       
        Line2D([0], [0],
               marker = "o",
               color = "none",
               markerfacecolor = shot_colours["miss"],
               markeredgecolor = "white",
               markersize = 20,
               label = "Miss"),
        
        Line2D([0], [0],
               marker = "o",
               color = "none",
               markerfacecolor = shot_colours["post"],
               markeredgecolor = "white",
               markersize=20,
               label="Post")]

    fig.legend(handles = legend_elements,
              loc = "lower center",
              ncol = 5, 
              bbox_to_anchor = (0.5,0.1),
              frameon = False,
              labelcolor = "#404040",
              fontsize = 16)

    # Plot sample xG values for scale
    xg_values = [0.1,0.3,0.5]

    xg_legend = [Line2D([0],[0],
                       marker = 'o',
                       color = "none",
                       markerfacecolor = "#808080",
                       markeredgecolor = "white",
                       markersize = np.sqrt(np.sqrt(xg) * 1000),
                       label =f"{xg:.1f} xG")
                    for xg in xg_values]

    fig.legend(handles = xg_legend,
              loc = "lower center",
              ncol = 3,
              bbox_to_anchor = (0.5,0.05),
              frameon = False,
              labelcolor = "#404040",
              fontsize = 16)

    # Annotate metrics on the figures
    # Home stats 
    ax[0].text(98,55, f"OPEN PLAY xG: {home_open_xg:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "left",
               va = "center")
    ax[0].text(98,52, f"SET PIECE xG: {home_set_piece_xg:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "left",
               va = "center")
    ax[0].text(2,55, f"xG / SHOT: {home_xg_per_shot:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "right",
               va = "center")
    ax[0].text(2,52, f"xGOT - xG: {home_qual:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "right",
               va = "center")
    # Away stats
    ax[1].text(98,55, f"OPEN PLAY xG: {away_open_xg:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "left",
               va = "center")
    ax[1].text(98,52, f"SET PIECE xG: {away_set_piece_xg:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "left",
               va = "center")
    ax[1].text(2,55, f"xG / SHOT: {away_xg_per_shot:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "right",
               va = "center")
    ax[1].text(2,52, f"xGOT - xG: {away_qual:.2f}",
               fontsize = 16,
               color = "#CCCCCC",
               ha = "right",
               va = "center")
    
    # Add titles
    ax[0].set_title(f"{home_team} Shot Map",
                   fontsize = 20,
                   color = "#2E2E2E",
                   pad = 10,
                   loc = "left")
    
    ax[1].set_title(f"{away_team} Shot Map",
                   fontsize = 20,
                   color = "#2E2E2E",
                   pad = 10,
                   loc = "left")

    # Save the figure
    fig.savefig(figures_folder / "shot_map.png",
                format = "png",
                dpi = 150,
                bbox_inches = "tight")

    plt.show()

    