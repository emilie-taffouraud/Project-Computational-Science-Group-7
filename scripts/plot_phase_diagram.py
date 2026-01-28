"""
Creates phase diagram by running simulations across (h,b) space

Color:
- Blue: Depolarizing (ΔNDI < -5)
- Gray: Persistent disagreement (-5 ≤ ΔNDI ≤ 5)
- Red: Polarizing (ΔNDI > 5)

theoretical critical boundary: b_c = 2/(h+1) 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_phase_diagram(
    summary_df: pd.DataFrame,
    outpath: str,
    title: str = "Phase Diagram of Opinion Dynamics",
) -> None:
    
    # 1. Check required columns
    needed = {"h", "b", "mean_delta_ndi"}
    missing = needed - set(summary_df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    plt.figure(figsize=(10, 7))
    
    # Flags to avoid duplicate legend entries
    depolarizing_plotted = False
    persistent_plotted = False
    polarizing_plotted = False
    
    # Iterate through rows and plot
    for _, row in summary_df.iterrows():
        delta_ndi = row['mean_delta_ndi']
        h = row['h']
        b = row['b']
        if np.isclose(h, 1.0) and np.isclose(b, 1.0):
            continue 
        # Blue :  Consensus
        if delta_ndi < 0:
            color = 'blue'
            marker = 'v'  # inverted triangle
            label = 'Depolarizing' if not depolarizing_plotted else ""
            depolarizing_plotted = True
            
        # Red :  Polarization
        else:
            color = 'red'
            marker = '^'  # triangle up
            label = 'Polarizing' if not polarizing_plotted else ""
            polarizing_plotted = True
            
        
        
        plt.scatter(h, b, c=color, s=200, marker=marker,
                   alpha=0.7, edgecolors='black', linewidths=1.5,
                   label=label, zorder=3)
    
    # Theoretical critical boundary b_c = 2 / (h + 1)

    h_min, h_max = summary_df['h'].min(), summary_df['h'].max()
    h_vals = np.linspace(h_min, h_max, 100)
    b_critical = 2.0 / (h_vals + 1)
    
    plt.plot(h_vals, b_critical, 'k--', linewidth=2,
            label='Theoretical boundary: $b_c = 2/(h+1)$', zorder=2)
    
    plt.xlabel('Homophily ($h = p_s/p_d$)', fontsize=12)
    plt.ylabel('Bias ($b$)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits
    plt.xlim(h_min - 0.5, h_max + 0.5)
    plt.ylim(-0.1, summary_df['b'].max() + 0.1)
    
    plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()

if __name__ == "__main__":
    input_csv = "results/summary.csv"
    output_png = "figures/phase_diagram_opinion.png"
    
    if os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        plot_phase_diagram(
            summary_df=df,
            outpath=output_png
        )