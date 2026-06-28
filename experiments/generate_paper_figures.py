#!/usr/bin/env python3
"""
Paper Figure Generation Script

Generate SCI paper required figures from experiment JSON data.
Target journal: IEEE IoTJ

Generated figures:
1. Fig05: SOTA Comparison (KeyGen/Encrypt/Decrypt/Total Time Comparison)
2. Fig06: Policy Depth Impact on Encrypt/Decrypt Time
3. Fig07: Communication Overhead (Ciphertext/Key Size + Transmission Time)
4. Fig08: Scalability (KeyGen/Encrypt/Decrypt Time with Attribute Count)
5. Fig09: Comprehensive Radar Chart (10 attributes, typical IoT scenario)
6. Fig10: Ablation Study Bar Chart

Author: AI Assistant
Date: 2026-06-04
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set academic style (English)
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# Color scheme (academic style)
COLORS = {
    'BSW07': '#2E86AB',  # Blue
    'BSW07-AND': '#A23B72',  # Magenta
    'OurScheme': '#F18F01',  # Orange
    'encrypt': '#2E86AB',  # Blue
    'decrypt': '#F18F01',  # Orange
    'keygen': '#2E86AB',  # Blue
    'ciphertext': '#2E86AB',  # Blue
    'key': '#F18F01',  # Orange
}

# Data file paths
RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = Path(__file__).parent.parent / 'paper' / 'figures'


def load_json(filename: str) -> dict:
    """Load JSON data file"""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_dir():
    """Create output directory"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {OUTPUT_DIR}")


def plot_fig1_sota_comparison():
    """
    Figure 1: SOTA Comparison (KeyGen/Encrypt/Decrypt Time Comparison)
    
    Three bar charts showing KeyGen, Encrypt, and Decrypt time.
    X-axis: Attribute Count (all 8: 10, 50, 100, 200, 500, 1000, 5000, 10000)
    Y-axis: Time (ms)
    """
    data = load_json('comparison/sota_comparison_table.json')
    
    # Extract data - use all 8 attribute counts
    # JSON key to display label mapping
    scheme_map = {
        'BSW07': 'BSW07',
        'LightweightABE': 'BSW07-AND',
        'OurScheme': 'T-CP-ABE (Ours)',
    }
    json_keys = list(scheme_map.keys())
    attr_counts = sorted(set(int(k.split('_')[-1]) for k in data.keys()))
    
    # Organize data (indexed by JSON key)
    keygen_data = {jk: [] for jk in json_keys}
    encrypt_data = {jk: [] for jk in json_keys}
    decrypt_data = {jk: [] for jk in json_keys}
    
    for attr in attr_counts:
        for jk in json_keys:
            key = f"{jk}_{attr}"
            if key in data:
                keygen_data[jk].append(data[key]['keygen_ms'])
                encrypt_data[jk].append(data[key]['encrypt_ms'])
                decrypt_data[jk].append(data[key]['decrypt_ms'])
    
    # Create figure - use 2x2 layout (legend at top center, avoid covering bars)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(attr_counts))
    width = 0.25
    
    # KeyGenTime
    ax = axes[0, 0]
    for i, jk in enumerate(json_keys):
        label = scheme_map[jk]
        ax.bar(x + i * width, keygen_data[jk], width, 
              label=label, color=list(COLORS.values())[i], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Key Generation Time')
    ax.set_xticks(x + width)
    ax.set_xticklabels(attr_counts, rotation=45)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # EncryptTime
    ax = axes[0, 1]
    for i, jk in enumerate(json_keys):
        label = scheme_map[jk]
        ax.bar(x + i * width, encrypt_data[jk], width, 
              label=label, color=list(COLORS.values())[i], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(b) Encryption Time (Fixed Policy)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(attr_counts, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # DecryptTime
    ax = axes[1, 0]
    for i, jk in enumerate(json_keys):
        label = scheme_map[jk]
        ax.bar(x + i * width, decrypt_data[jk], width, 
              label=label, color=list(COLORS.values())[i], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(c) Decryption Time (Fixed Policy)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(attr_counts, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Total Time
    ax = axes[1, 1]
    total_data = {jk: [k + e + d for k, e, d in zip(keygen_data[jk], encrypt_data[jk], decrypt_data[jk])] 
                  for jk in json_keys}
    for i, jk in enumerate(json_keys):
        label = scheme_map[jk]
        ax.bar(x + i * width, total_data[jk], width, 
              label=label, color=list(COLORS.values())[i], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(d) Total Time')
    ax.set_xticks(x + width)
    ax.set_xticklabels(attr_counts, rotation=45)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Shared legend at top center (get all handles from first subplot)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
              bbox_to_anchor=(0.5, 1.02), framealpha=0.9, edgecolor='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / 'fig05_sota_comparison.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig05_sota_comparison.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig05 SOTA Comparison saved: {output_path}")
    plt.close()


def plot_fig2_policy_depth():
    """
    Figure 2: Policy Depth Impact on Encrypt/Decrypt Time
    
    Dual Y-axis or dual subplot showing policy depth impact on encrypt and decrypt time.
    X-axis: Policy Depth (2, 5, 8, 10, 15, 20)
    Y-axis: Time (ms)
    """
    enc_data = load_json('scalability/encryption_scalability.json')
    dec_data = load_json('scalability/decryption_scalability.json')
    
    # Extract policy depth data
    depths = sorted([int(d) for d in enc_data['by_depth'].keys()])
    enc_means = [enc_data['by_depth'][str(d)]['mean_ms'] for d in depths]
    enc_stds = [enc_data['by_depth'][str(d)]['std_ms'] for d in depths]
    dec_means = [dec_data['by_depth'][str(d)]['mean_ms'] for d in depths]
    dec_stds = [dec_data['by_depth'][str(d)]['std_ms'] for d in depths]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # EncryptTime
    ax.errorbar(depths, enc_means, yerr=enc_stds, 
               marker='o', markersize=8, linewidth=2, capsize=5,
               label='Encryption', color=COLORS['encrypt'], alpha=0.85)
    
    # DecryptTime
    ax.errorbar(depths, dec_means, yerr=dec_stds, 
               marker='s', markersize=8, linewidth=2, capsize=5,
               label='Decryption', color=COLORS['decrypt'], alpha=0.85)
    
    ax.set_xlabel('Policy Depth (Number of AND Gates)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Impact of Policy Depth on Encryption/Decryption Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depths)
    
    # Add trend line
    z_enc = np.polyfit(depths, enc_means, 1)
    p_enc = np.poly1d(z_enc)
    ax.plot(depths, p_enc(depths), '--', color=COLORS['encrypt'], alpha=0.5, linewidth=1)
    
    z_dec = np.polyfit(depths, dec_means, 1)
    p_dec = np.poly1d(z_dec)
    ax.plot(depths, p_dec(depths), '--', color=COLORS['decrypt'], alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig06_policy_depth_impact.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig06_policy_depth_impact.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig06 Policy Depth Impact saved: {output_path}")
    plt.close()


def plot_fig3_communication_overhead():
    """
    Figure 3: Communication Overhead (Ciphertext/Key Size + Transmission Time)
    
    Dual subplot: left shows ciphertext/key size, right shows transmission time.
    X-axis: Attribute Count
    Y-axis: Size (KB) or Time (ms)
    """
    data = load_json('communication/communication_overhead_table.json')
    
    # Extract data
    attr_counts = sorted([int(k) for k in data.keys()])
    ct_sizes = [data[str(a)]['ct_size_kb'] for a in attr_counts]
    sk_sizes = [data[str(a)]['sk_size_kb'] for a in attr_counts]
    time_4g = [data[str(a)]['time_4g_ms'] for a in attr_counts]
    time_5g = [data[str(a)]['time_5g_ms'] for a in attr_counts]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Ciphertext/Key Size
    ax1.plot(attr_counts, ct_sizes, marker='o', markersize=8, linewidth=2, 
            label='Ciphertext', color=COLORS['ciphertext'], alpha=0.85)
    ax1.plot(attr_counts, sk_sizes, marker='s', markersize=8, linewidth=2, 
            label='Secret Key', color=COLORS['key'], alpha=0.85)
    ax1.set_xlabel('Number of Attributes')
    ax1.set_ylabel('Size (KB)')
    ax1.set_title('Ciphertext and Key Size')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Right subplot: Transmission Time
    ax2.plot(attr_counts, time_4g, marker='o', markersize=8, linewidth=2, 
            label='4G Network', color=COLORS['ciphertext'], alpha=0.85)
    ax2.plot(attr_counts, time_5g, marker='s', markersize=8, linewidth=2, 
            label='5G Network', color=COLORS['key'], alpha=0.85)
    ax2.set_xlabel('Number of Attributes')
    ax2.set_ylabel('Transmission Time (ms)')
    ax2.set_title('Network Transmission Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig07_communication_overhead.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig07_communication_overhead.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig07 Communication Overhead saved: {output_path}")
    plt.close()


def plot_fig4_scalability():
    """
    Figure 4: Scalability (KeyGen/Encrypt/Decrypt Time with Attribute Count)
    
    Three subplots showing KeyGen, Encrypt, and Decrypt time scalability.
    X-axis: Attribute Count (logarithmic scale)
    Y-axis: Time (ms)
    """
    data = load_json('scalability/scalability_summary_table.json')
    
    # Extract data
    attr_counts = sorted([int(k) for k in data.keys()])
    keygen_times = [data[str(a)]['keygen_ms'] for a in attr_counts]
    encrypt_times = [data[str(a)]['encrypt_ms'] for a in attr_counts]
    decrypt_times = [data[str(a)]['decrypt_ms'] for a in attr_counts]
    throughput = [data[str(a)]['throughput_ops'] for a in attr_counts]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # KeyGenTime
    ax = axes[0, 0]
    ax.plot(attr_counts, keygen_times, marker='o', markersize=8, linewidth=2, 
            color=COLORS['keygen'], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Key Generation Time')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add linear fit
    z = np.polyfit(np.log10(attr_counts), np.log10(keygen_times), 1)
    p = np.poly1d(z)
    ax.plot(attr_counts, 10**p(np.log10(attr_counts)), '--', 
            color=COLORS['keygen'], alpha=0.5, linewidth=1,
            label=f'Slope: {z[0]:.2f}')
    ax.legend()
    
    # EncryptTime
    ax = axes[0, 1]
    ax.plot(attr_counts, encrypt_times, marker='s', markersize=8, linewidth=2, 
            color=COLORS['encrypt'], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Encryption Time')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # DecryptTime
    ax = axes[1, 0]
    ax.plot(attr_counts, decrypt_times, marker='^', markersize=8, linewidth=2, 
            color=COLORS['decrypt'], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Decryption Time')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Throughput
    ax = axes[1, 1]
    ax.bar(range(len(attr_counts)), throughput, color=COLORS['OurScheme'], alpha=0.85)
    ax.set_xlabel('Number of Attributes')
    ax.set_ylabel('Throughput (ops/s)')
    ax.set_title('System Throughput')
    ax.set_xticks(range(len(attr_counts)))
    ax.set_xticklabels(attr_counts, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig08_scalability.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig08_scalability.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig08 Scalability saved: {output_path}")
    plt.close()


def plot_fig5_comprehensive_comparison():
    """
    Figure 5: Comprehensive Comparison Radar Chart
    
    Show comparison of three schemes across different dimensions.
    Use 10 attribute data (typical IoT scenario), add comments explaining functional differences.
    """
    data = load_json('comparison/sota_comparison_table.json')
    
    # Use 10 attribute data (typical IoT scenario)
    attr = 10
    scheme_map = {'BSW07': 'BSW07', 'LightweightABE': 'BSW07-AND', 'OurScheme': 'T-CP-ABE (Ours)'}
    json_keys = list(scheme_map.keys())
    
    # Extract data
    keygen_times = [data[f'{s}_{attr}']['keygen_ms'] for s in json_keys]
    encrypt_times = [data[f'{s}_{attr}']['encrypt_ms'] for s in json_keys]
    decrypt_times = [data[f'{s}_{attr}']['decrypt_ms'] for s in json_keys]
    
    # Normalize to 0-1 range (lower is better)
    def normalize(values):
        max_val = max(values)
        return [v / max_val for v in values]
    
    keygen_norm = normalize(keygen_times)
    encrypt_norm = normalize(encrypt_times)
    decrypt_norm = normalize(decrypt_times)
    
    # Create radar chart
    categories = ['KeyGen', 'Encrypt', 'Decrypt']
    N = len(categories)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw each scheme
    for i, jk in enumerate(json_keys):
        values = [keygen_norm[i], encrypt_norm[i], decrypt_norm[i]]
        values += values[:1]  # Close the loop
        
        label = scheme_map[jk]
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=label, color=list(COLORS.values())[i], alpha=0.85)
        ax.fill(angles, values, alpha=0.15, color=list(COLORS.values())[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Performance Comparison ({attr} Attributes)\n(Lower is Better)', 
                size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add comments explaining functional differences
    note_text = (
        "Note: T-CP-ABE provides additional features:\n"
        "• Time-aware predicates\n"
        "• THRESHOLD gate support\n"
        "• Attribute revocation\n"
        "• BLS authentication\n"
        "• Diffusion-based threat detection"
    )
    fig.text(0.02, 0.02, note_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig09_radar_comparison.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig09_radar_comparison.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig09 Radar Comparison saved: {output_path}")
    plt.close()


def plot_fig6_ablation_study():
    """
    Figure 6: Ablation Study Bar Chart
    
    Show contribution of each component to performance.
    """
    data = load_json('ablation/full_ablation_study.json')
    
    # Configuration names and colors
    configs = ['Full', 'NoTimePredicate', 'NoCache', 'NoSubprocess', 'NoDiffusion', 'NoDigitalTwin', 'Minimal']
    config_labels = ['Full', 'w/o Time', 'w/o Cache', 'w/o Subprocess', 'w/o Diffusion', 'w/o DT', 'Minimal']
    
    # Extract data
    keygen_times = [data[c]['keygen_time'] * 1000 for c in configs]  # Convert to ms
    encrypt_times = [data[c]['encrypt_time'] * 1000 for c in configs]
    decrypt_times = [data[c]['decrypt_time'] * 1000 for c in configs]
    total_times = [data[c]['total_time'] * 1000 for c in configs]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(configs))
    width = 0.6
    
    # KeyGenTime
    ax = axes[0, 0]
    bars = ax.bar(x, keygen_times, width, color='#2E86AB', alpha=0.85)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Key Generation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, keygen_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # EncryptTime
    ax = axes[0, 1]
    bars = ax.bar(x, encrypt_times, width, color='#A23B72', alpha=0.85)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(b) Encryption Time')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, encrypt_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # DecryptTime
    ax = axes[1, 0]
    bars = ax.bar(x, decrypt_times, width, color='#F18F01', alpha=0.85)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(c) Decryption Time')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, decrypt_times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Total Time
    ax = axes[1, 1]
    bars = ax.bar(x, total_times, width, color='#6C757D', alpha=0.85)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('(d) Total Time')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and percentage
    full_total = total_times[0]
    for bar, val in zip(bars, total_times):
        pct = (val / full_total - 1) * 100
        label = f'{val:.1f}'
        if pct != 0:
            label += f'\n({pct:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'fig10_ablation_study.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    output_path_png = OUTPUT_DIR / 'fig10_ablation_study.png'
    plt.savefig(output_path_png, bbox_inches='tight')
    print(f"✅ Fig10 Ablation Study saved: {output_path}")
    plt.close()


def generate_latex_table():
    """
    Generate LaTeX format table code
    """
    # SOTA Comparison table
    data = load_json('comparison/sota_comparison_table.json')
    
    latex_code = """
% Table 1: SOTA Comparison Experiment Results
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison with State-of-the-Art Schemes}
\\label{tab:sota_comparison}
\\begin{tabular}{l|c|c|c|c}
\\hline
\\textbf{Scheme} & \\textbf{Attrs} & \\textbf{KeyGen (ms)} & \\textbf{Encrypt (ms)} & \\textbf{Decrypt (ms)} \\\\
\\hline
"""
    
    scheme_map = {'BSW07': 'BSW07', 'LightweightABE': 'BSW07-AND', 'OurScheme': 'T-CP-ABE (Ours)'}
    json_keys = list(scheme_map.keys())
    attr_counts = sorted(set(int(k.split('_')[-1]) for k in data.keys()))
    
    for attr in attr_counts:
        for i, jk in enumerate(json_keys):
            key = f"{jk}_{attr}"
            if key in data:
                d = data[key]
                label = scheme_map[jk]
                if i == 0:
                    latex_code += f"{label} & {attr} & {d['keygen_ms']:.2f} & {d['encrypt_ms']:.2f} & {d['decrypt_ms']:.2f} \\\\\n"
                else:
                    latex_code += f" & {attr} & {d['keygen_ms']:.2f} & {d['encrypt_ms']:.2f} & {d['decrypt_ms']:.2f} \\\\\n"
        if attr != attr_counts[-1]:
            latex_code += "\\hline\n"
    
    latex_code += """\\hline
\\end{tabular}
\\end{table}

% Note: Encrypt and Decrypt times use fixed policy structure (4 AND gates), so they don't vary with attribute count.
"""
    
    # Ablation Study table
    ablation_data = load_json('ablation/full_ablation_study.json')
    
    latex_code += """
% Table 2: Ablation Study Results
\\begin{table}[htbp]
\\centering
\\caption{Ablation Study Results}
\\label{tab:ablation}
\\begin{tabular}{l|c|c|c|c}
\\hline
\\textbf{Configuration} & \\textbf{KeyGen (ms)} & \\textbf{Encrypt (ms)} & \\textbf{Decrypt (ms)} & \\textbf{Total (ms)} \\\\
\\hline
"""
    
    configs = ['Full', 'NoTimePredicate', 'NoCache', 'NoSubprocess', 'NoDiffusion', 'NoDigitalTwin', 'Minimal']
    config_labels = ['Full', 'w/o Time Predicate', 'w/o Cache', 'w/o Subprocess', 'w/o Diffusion', 'w/o Digital Twin', 'Minimal']
    
    for i, (config, label) in enumerate(zip(configs, config_labels)):
        d = ablation_data[config]
        latex_code += f"{label} & {d['keygen_time']*1000:.1f} & {d['encrypt_time']*1000:.1f} & {d['decrypt_time']*1000:.1f} & {d['total_time']*1000:.1f} \\\\\n"
        if i == 0:
            latex_code += "\\hline\n"
    
    latex_code += """\\hline
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX code
    latex_path = OUTPUT_DIR / 'latex_tables.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"✅ LaTeX tables saved: {latex_path}")


def main():
    """Main function"""
    print("=" * 60)
    print("Paper Figure Generation Script")
    print("=" * 60)
    
    # Create output directory
    create_output_dir()
    
    # Generate figures
    print("\n📊 Generating figures...")
    plot_fig1_sota_comparison()      # Fig05
    plot_fig2_policy_depth()         # Fig06
    plot_fig3_communication_overhead() # Fig07
    plot_fig4_scalability()          # Fig08
    plot_fig5_comprehensive_comparison() # Fig09
    plot_fig6_ablation_study()       # Fig10
    
    # Generate LaTeX tables
    print("\n📝 Generating LaTeX tables...")
    generate_latex_table()
    
    print("\n" + "=" * 60)
    print("✅ All figures and tables generated successfully!")
    print(f"📁 OutputTable of Contents: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
