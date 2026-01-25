import matplotlib.pyplot as plt
import numpy as np
import os

from src.network import build_two_island
from src.models import step_degroot, step_biased
'''
Visualizes the temporal evolution of average opinions for two islands.
'''
def main():
    N_USERS = 100        
    N_ISLAND = N_USERS // 2
    T = 60              
    PD_VAL = 0.04        
    
    conditions = [
        {
            'title': 'Consensus',
            'subtitle': r'Low Bias ($b=0.0$)',
            'h': 8, 'ps': 0.08, 'b': 0.0, 
            'model': 'degroot'
        },
        {
            'title': 'Persistent Disagreement', 
            'subtitle': r'Medium Bias ($b=0.3$)', 
            'h': 8, 'ps': 0.80, 'b': 0.3,
            'model': 'biased'
        },
        {
            'title': 'Polarization ',
            'subtitle': r'High Bias ($b=1.0$)',
            'h': 8, 'ps': 0.80, 'b': 1.0,
            'model': 'biased'
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    plt.subplots_adjust(wspace=0.15) 
    # simulation
    for idx, cond in enumerate(conditions):
        ax = axes[idx]
        
        adj, _ = build_two_island(n=N_ISLAND, ps=cond['ps'], pd=PD_VAL)
        
        x = [0.7] * N_ISLAND + [0.3] * N_ISLAND

        mean_1_history = [0.7]
        mean_2_history = [0.3]
        
        for t in range(1, T + 1):
            if cond['model'] == 'degroot':
                x = step_degroot(x, adj, w_self=0.0)
            else:
                x = step_biased(x, adj, b=cond['b'], w_self=0.0)
            
            m1 = sum(x[:N_ISLAND]) / N_ISLAND
            m2 = sum(x[N_ISLAND:]) / N_ISLAND
            
            mean_1_history.append(m1)
            mean_2_history.append(m2)

        time_steps = range(T + 1)
        
        ax.plot(time_steps, mean_1_history, color='#D62728', linewidth=3, label='Island 1 Mean')
        ax.plot(time_steps, mean_2_history, color='#1F77B4', linewidth=3, label='Island 2 Mean')
        
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel("Time Step ($t$)", fontsize=14)
        ax.set_title(f"{cond['title']}\n{cond['subtitle']}", fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        
        if idx == 0: 
            ax.set_ylabel("Average Opinion ($\overline{x}$)", fontsize=14)
            ax.legend(loc='lower right', fontsize=12)
            ax.annotate('Converged', xy=(T-5, 0.5), xytext=(T-20, 0.65),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontsize=14, fontweight='bold') # <--- 统一大小粗细

        elif idx == 1: 
            ax.annotate('Stable Gap', 
                        xy=(T-5, mean_1_history[-1]), 
                        xytext=(T/2, 0.5),
                        arrowprops=dict(facecolor='purple', arrowstyle='-|>', connectionstyle="arc3,rad=-0.2"),
                        color='purple', ha='center',
                        fontsize=14, fontweight='bold') # <--- 统一大小粗细

        elif idx == 2: 
            ax.text(T-2, 0.925, 'Extreme', color='#D62728', ha='right', 
                    fontsize=14, fontweight='bold') # <--- 统一大小粗细
            ax.text(T-2, 0.05, 'Extreme', color='#1F77B4', ha='right', 
                    fontsize=14, fontweight='bold') # <--- 统一大小粗细


    os.makedirs("figures", exist_ok=True)
    out_path = "figures/island_trends.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()