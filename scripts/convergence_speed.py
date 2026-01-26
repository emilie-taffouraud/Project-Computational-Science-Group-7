"""
Visualize convergence speed
This codes generates a time-series visualization showing the evolution of the Network Disagreement Index (NDI) over time (t).
It compares convergence speeds with different values of Homophily (h) and Bias (b).
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from src.network import build_two_island
from src.models import step_biased, step_degroot
from src.metrics import ndi


def plot_convergence_analysis(
    outpath: str = "figures/convergence_speed.png",
    n: int = 100,
    T: int = 100,
    pd_value: float = 0.10,
) -> None:
    
    # Define the conditions we want to plot
    conditions_to_plot = [
        (0.0, 1),
        (0.0, 8),
        (0.5, 1), 
        (0.5, 8),
        (1.0, 1),
        (1.0, 8)
    ]
    
    cmap = plt.get_cmap('tab10')
    # Assign a unique color to each of these 6 conditions
    color_map = {
        cond: cmap(i) for i, cond in enumerate(conditions_to_plot)
    }
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    all_histories = [] 
    
    for (b, h) in conditions_to_plot:
        ps = h * pd_value
        adj, edges = build_two_island(n=n, ps=ps, pd=pd_value)
        x = [0.7] * n + [0.3] * n
        
        ndi_history = []
        
        for t in range(T + 1):
            curr_ndi = ndi(x, edges)
            ndi_history.append(curr_ndi)
            
            if t < T:
                if b == 0:
                    x = step_degroot(x, adj, w_self=0.0)
                else:
                    x = step_biased(x, adj, b=b, w_self=0.0)
        
        all_histories.append(ndi_history)
        
        delta_ndi = ndi_history[-1] - ndi_history[0]
        
        # Lines with $b>0$ are dotted.
        # lines with $b=0$ are solid.
        if b > 0:
            linestyle = (0, (1, 2)) 
        else:
            linestyle = '-'        
        
        color = color_map[(b, h)]
        
        label = f'$b={b:.1f}, h={h}$ (Δ={delta_ndi:+.1f})'
        
        ax1.plot(range(T + 1), ndi_history, 
                 color=color, 
                 linestyle=linestyle, 
                 linewidth=3.0, 
                 label=label, 
                 alpha=0.9)

    ax1.set_xlabel('Time Step ($t$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Network Disagreement Index (NDI)', fontsize=13, fontweight='bold')
    ax1.set_title('NDI Evolution: Unique Colors per Condition', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Main Legend 
    ax1.legend(fontsize=11, ncol=2, loc='best', framealpha=0.95)
    
    ax1.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # Scaling
    start_vals = [h[0] for h in all_histories]
    end_vals = [h[-1] for h in all_histories]
    
    has_decrease = any(e < s * 0.5 for s, e in zip(start_vals, end_vals))
    has_increase = any(e > s * 1.2 for s, e in zip(start_vals, end_vals))
    
    if has_decrease and not has_increase:
        ax1.set_yscale('log')
        ax1.set_ylabel('NDI (log scale)', fontsize=13, fontweight='bold')
    elif has_increase and has_decrease:
        ax1.set_yscale('symlog', linthresh=1.0)
        ax1.set_ylabel('NDI (symlog scale)', fontsize=13, fontweight='bold')

    ax1.text(0.98, 0.02,
            'Solid Line: $b=0$\nDotted Line: $b>0$',
            transform=ax1.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_convergence_analysis()