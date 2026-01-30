from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from src.network import build_two_island
from src.models import step_degroot, step_biased
'''
Opinion Evolution Visualization

Experimental Conditions:
1. The first two rows examine the influence of homophily (h).
2. The last two rows examine the influence of confirmation bias (b).

Polarization Metric:
Calculates the Euclidean distance between the centers of two islands
A larger distance implies Higher polarization.

'''
def visualize_opinion_evolution_clean(
    outpath: str = "figures/opinion_evolution_grid.png",
    n: int = 100,
    T: int = 50,
    pd_value: float = 0.10,
) -> None:
    
    # define four conditions
    conditions = [
        {
            'name': 'Consensus (Low Homophily)\n$h=1, b=0.0$',
            'h': 1, 'ps': 0.10, 'b': 0.0, 
            'model': 'degroot'
        },
        {
            'name': 'Consensus (High Homophily)\n$h=8, b=0.0$',
            'h': 8, 'ps': 0.80, 'b': 0.0, 
            'model': 'degroot'
        },
        {
            'name': 'Persistent Disagreement\n$h=8, b=0.3$',
            'h': 8, 'ps': 0.80, 'b': 0.3,
            'model': 'biased'
        },
        {
            'name': 'Polarization\n$h=8, b=1.0$',
            'h': 8, 'ps': 0.80, 'b': 1.0,
            'model': 'biased'
        }
    ]
    
    # Create 4 rows x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    
    for row_idx, cond in enumerate(conditions):
        adj, _ = build_two_island(n=n, ps=cond['ps'], pd=pd_value)
        # initialization
        x = [0.7] * n + [0.3] * n
        
        # running the simulation
        snapshots = {0: list(x)}
        for t in range(1, T + 1):
            if cond['model'] == 'degroot':
                x = step_degroot(x, adj, w_self=0.0)
            else:
                x = step_biased(x, adj, b=cond['b'], w_self=0.0)
            
            # record intermediate and final states
            if t == T // 2:
                snapshots[T // 2] = list(x)
            elif t == T:
                snapshots[T] = list(x)
        
        time_points = [0, T // 2, T]
        col_titles = ['Initial\n$t=0$', 'Middle\n$t=25$', 'Final\n$t=50$']
        
        for col_idx, (time_point, title) in enumerate(zip(time_points, col_titles)):
            ax = axes[row_idx, col_idx]
            opinions = np.array(snapshots[time_point])
            
            np.random.seed(42 + time_point) 
            
            i1 = opinions[:n]
            i2 = opinions[n:]
            
            sep_factor = 5.0
            x1_pos = (i1 - 0.5) * sep_factor
            x2_pos = (i2 - 0.5) * sep_factor
            y1_pos = np.random.randn(n) * 0.5
            y2_pos = np.random.randn(n) * 0.5
            
            all_x = np.concatenate([x1_pos, x2_pos])
            all_y = np.concatenate([y1_pos, y2_pos])
            
            try:
                positions = np.vstack([all_x, all_y])
                kernel = gaussian_kde(positions, bw_method=0.15)
                
                xg = np.linspace(all_x.min()-1, all_x.max()+1, 100)
                yg = np.linspace(all_y.min()-1, all_y.max()+1, 100)
                X, Y = np.meshgrid(xg, yg)
                grid_pos = np.vstack([X.ravel(), Y.ravel()])
                Z = kernel(grid_pos).reshape(X.shape)
                
                ax.imshow(Z, extent=[xg.min(), xg.max(), yg.min(), yg.max()],
                          origin='lower', cmap='Blues', alpha=0.6, aspect='auto')
            except:
                pass 
            
            ax.scatter(x1_pos, y1_pos, c='royalblue', s=15, alpha=0.6, edgecolors='none')
            ax.scatter(x2_pos, y2_pos, c='orangered', s=15, alpha=0.6, edgecolors='none')
            
            # draw the center of the group
            c1x, c1y = np.mean(x1_pos), np.mean(y1_pos)
            c2x, c2y = np.mean(x2_pos), np.mean(y2_pos)
            ax.scatter([c1x], [c1y], c='blue', s=300, marker='*', edgecolors='white', linewidth=0.5)
            ax.scatter([c2x], [c2y], c='red', s=300, marker='*', edgecolors='white', linewidth=0.5)
            
            # Euclidean distance between cluster centers
            dist = np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)
            ax.text(0.95, 0.95, f'Dist: {dist:.1f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=15, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if row_idx == 0:
                ax.set_title(title, fontweight='bold', fontsize=15)
            if col_idx == 0:
                ax.set_ylabel(cond['name'], fontweight='bold', fontsize=12)
                
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xticks([]) 
            ax.set_yticks([])
            
            if row_idx == 0 and col_idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='Island 1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered', label='Island 2')
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=15)

    plt.suptitle('Opinion Evolution', fontsize=20, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200) 
    plt.close()

if __name__ == "__main__":
    visualize_opinion_evolution_clean()

