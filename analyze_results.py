#!/usr/bin/env python3
"""
LLM Benchmark Analysis Script v2 - Interactive Version
Software Quality Engineering 2025/2026 - Homework 3

Authors:
- Giovanni Altieri (309006)
- Matteo Patella (308056)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import glob
from datetime import datetime

try:
    from simple_term_menu import TerminalMenu
    HAS_MENU = True
except ImportError:
    HAS_MENU = False

DEFAULT_INPUT_DIR = "benchmark_results"
DEFAULT_OUTPUT_SUBDIR = "results_analyzed"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


METRICS_DOC = """
================================================================================
                        GUIDA ALLE METRICHE DI BENCHMARK
================================================================================

1. TIME TO FIRST TOKEN (TTFT)
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo tra invio richiesta e ricezione del primo token.
   CALCOLO: TTFT = load_duration + prompt_eval_duration
   INTERPRETAZIONE: Valori bassi = risposta più reattiva
   UNITÀ: secondi (s)

2. INTER-TOKEN LATENCY (ITL)
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo medio tra generazione di token consecutivi.
   CALCOLO: ITL = eval_duration / (eval_count - 1)
   INTERPRETAZIONE: Valori bassi = generazione più fluida
   UNITÀ: secondi (s)

3. TIME PER OUTPUT TOKEN (TPOT)
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo medio per generare ogni token di output.
   CALCOLO: TPOT = eval_duration / eval_count
   DIFFERENZA CON ITL: TPOT divide per n, ITL divide per n-1
   UNITÀ: secondi (s)

4. END-TO-END LATENCY (E2E)
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo totale dall'invio alla ricezione dell'ultimo token.
   CALCOLO: E2E = timestamp_fine - timestamp_inizio
   FORMULA: E2E ≈ TTFT + (output_tokens × TPOT)
   UNITÀ: secondi (s)

5. THROUGHPUT
--------------------------------------------------------------------------------
   DEFINIZIONE: Token generati per secondo.
   CALCOLO: Throughput = output_tokens / generation_time
   INTERPRETAZIONE: Valori alti = generazione più veloce
   UNITÀ: token/secondo (tok/s)

6. PROMPT PROCESSING TIME
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo per processare il prompt di input.
   CALCOLO: prompt_eval_duration da Ollama
   INTERPRETAZIONE: Cresce con lunghezza prompt
   UNITÀ: secondi (s)

7. GENERATION TIME
--------------------------------------------------------------------------------
   DEFINIZIONE: Tempo puro di generazione token.
   CALCOLO: eval_duration da Ollama
   RELAZIONE: Generation_Time = output_tokens × TPOT
   UNITÀ: secondi (s)

8. COEFFICIENT OF VARIATION (CV)
--------------------------------------------------------------------------------
   DEFINIZIONE: Variabilità relativa (stabilità risultati).
   CALCOLO: CV = deviazione_standard / media
   INTERPRETAZIONE:
     CV < 0.1: Molto stabile
     CV 0.1-0.3: Moderato
     CV > 0.3: Alta variabilità
   UNITÀ: adimensionale

9. CORRELAZIONE (r di Pearson)
--------------------------------------------------------------------------------
   DEFINIZIONE: Forza relazione lineare tra variabili.
   RANGE: -1 (negativa) a +1 (positiva)
   INTERPRETAZIONE:
     |r| > 0.5: correlazione forte (***)
     |r| > 0.3: correlazione moderata (**)
     |r| > 0.1: correlazione debole (*)

================================================================================
"""


def find_benchmark_files(directory):
    patterns = [
        os.path.join(directory, "benchmark_results_*.csv"),
        os.path.join(directory, "benchmark_results_*.jsonl"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def select_file_interactive(files):
    if not files:
        return None
    
    display_names = []
    for f in files:
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size = os.path.getsize(f) / 1024
        name = os.path.basename(f)
        display_names.append(f"{name} ({mtime.strftime('%Y-%m-%d %H:%M')}, {size:.1f} KB)")
    
    if HAS_MENU:
        terminal_menu = TerminalMenu(display_names, title="Seleziona file:")
        idx = terminal_menu.show()
        return files[idx] if idx is not None else None
    else:
        print("File disponibili:")
        for i, name in enumerate(display_names, 1):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = input(f"\nSeleziona numero [1]: ").strip()
                if choice == "":
                    return files[0]
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
            except ValueError:
                print_error("Numero non valido")


def configure_analysis_interactive():
    print_header("CONFIGURAZIONE ANALISI")
    config = {}
    
    print(f"{Colors.BOLD}1. SELEZIONE FILE INPUT{Colors.END}\n")
    print_info(f"Directory predefinita: {DEFAULT_INPUT_DIR}")
    
    custom_path = input(f"\nPercorso [{Colors.CYAN}{DEFAULT_INPUT_DIR}{Colors.END}]: ").strip()
    search_dir = custom_path if custom_path else DEFAULT_INPUT_DIR
    
    if os.path.isfile(search_dir):
        config['input_file'] = search_dir
    elif os.path.isdir(search_dir):
        files = find_benchmark_files(search_dir)
        if not files:
            print_error(f"Nessun file trovato in: {search_dir}")
            sys.exit(1)
        print_info(f"Trovati {len(files)} file")
        config['input_file'] = select_file_interactive(files)
        if not config['input_file']:
            sys.exit(1)
    else:
        print_error(f"Percorso non trovato: {search_dir}")
        sys.exit(1)
    
    print_success(f"File: {os.path.basename(config['input_file'])}")
    
    print(f"\n{Colors.BOLD}2. DIRECTORY OUTPUT{Colors.END}\n")
    input_dir = os.path.dirname(config['input_file']) or "."
    default_output = os.path.join(input_dir, DEFAULT_OUTPUT_SUBDIR)
    
    custom_output = input(f"Directory output [{Colors.CYAN}{default_output}{Colors.END}]: ").strip()
    config['output_dir'] = custom_output if custom_output else default_output
    
    return config


def load_results(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.jsonl'):
        import jsonlines
        with jsonlines.open(filepath, 'r') as reader:
            return pd.DataFrame(list(reader))
    raise ValueError(f"Formato non supportato: {filepath}")


def summary_stats(data):
    mu = data.mean()
    sigma = data.std(ddof=1) if len(data) > 1 else 0.0
    std_error = sigma / np.sqrt(len(data)) if len(data) > 1 else 0.0
    return {
        "mean": mu, "median": data.median(), "std": sigma,
        "min": data.min(), "max": data.max(),
        "cv": sigma / mu if mu != 0 else np.nan,
        "ci_95_low": mu - 1.96 * std_error,
        "ci_95_high": mu + 1.96 * std_error
    }


def analyze_prompt_length(df, output_dir):
    print_info("Analisi: Prompt Length vs Metriche")
    models = df['model'].unique()
    correlations = {}
    
    for model in models:
        model_data = df[df['model'] == model]
        correlations[model] = {}
        for metric in ['ttft', 'e2e_latency', 'throughput']:
            if metric in df.columns:
                correlations[model][metric] = model_data['prompt_char_length'].corr(model_data[metric])
    
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model]
        axes[idx].scatter(model_data['prompt_char_length'], model_data['ttft'], alpha=0.5, s=20, color='#E57373')
        z = np.polyfit(model_data['prompt_char_length'], model_data['ttft'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(model_data['prompt_char_length'].min(), model_data['prompt_char_length'].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
        corr = correlations[model].get('ttft', 0)
        axes[idx].set_title(f"{model}\nr = {corr:.3f}")
        axes[idx].set_xlabel("Prompt Length (chars)")
        axes[idx].set_ylabel("TTFT (s)")
        axes[idx].legend()
    
    plt.suptitle("Time To First Token vs Prompt Length", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ttft_vs_prompt_length.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: ttft_vs_prompt_length.png")
    return correlations


def analyze_output_length(df, output_dir):
    print_info("Analisi: Output Length vs Metriche")
    models = df['model'].unique()
    correlations = {}
    
    for model in models:
        model_data = df[df['model'] == model]
        correlations[model] = {'e2e_latency': model_data['output_tokens'].corr(model_data['e2e_latency'])}
    
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model]
        axes[idx].scatter(model_data['output_tokens'], model_data['e2e_latency'], alpha=0.5, s=20, color='#E57373')
        z = np.polyfit(model_data['output_tokens'], model_data['e2e_latency'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(model_data['output_tokens'].min(), model_data['output_tokens'].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')
        corr = correlations[model]['e2e_latency']
        axes[idx].set_title(f"{model}\nr = {corr:.3f}")
        axes[idx].set_xlabel("Output Tokens")
        axes[idx].set_ylabel("E2E Latency (s)")
        axes[idx].legend()
    
    plt.suptitle("End-to-End Latency vs Output Length", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/e2e_vs_output_length.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: e2e_vs_output_length.png")
    return correlations


def analyze_model_comparison(df, output_dir):
    print_info("Analisi: Confronto Modelli")
    
    metrics = ['ttft', 'itl', 'e2e_latency', 'throughput']
    metrics = [m for m in metrics if m in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, metric in enumerate(metrics[:4]):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=metric, by='model', ax=ax)
        ax.set_title(metric.upper())
        ax.set_xlabel("Model")
    plt.suptitle("Metric Distribution by Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: model_comparison_boxplot.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, metric in enumerate(metrics[:4]):
        ax = axes[idx // 2, idx % 2]
        sns.violinplot(data=df, x='model', y=metric, ax=ax)
        ax.set_title(metric.upper())
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle("Metric Distributions (Violin)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_violin.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: model_comparison_violin.png")
    
    summary = df.groupby('model').agg({m: 'mean' for m in metrics}).round(4)
    norm = summary.copy()
    for col in norm.columns:
        if col in ['ttft', 'e2e_latency', 'itl']:
            norm[col] = norm[col].min() / norm[col]
        else:
            norm[col] = norm[col] / norm[col].max()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(norm, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, vmin=0.6, vmax=1.0)
    ax.set_title('Model Performance (Normalized, higher=better)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: model_comparison_heatmap.png")


def analyze_variability(df, output_dir):
    print_info("Analisi: Variabilità (CV)")
    models = df['model'].unique()
    metrics = ['e2e_latency', 'ttft', 'throughput']
    
    variability = df.groupby(['model', 'prompt_id']).agg({m: ['mean', 'std'] for m in metrics if m in df.columns})
    for metric in metrics:
        if metric in df.columns:
            variability[(metric, 'cv')] = variability[(metric, 'std')] / variability[(metric, 'mean')]
    
    cv_data = []
    for model in models:
        row = {'Model': model}
        for metric in metrics:
            if metric in df.columns:
                row[f'{metric}_cv'] = variability.loc[model][(metric, 'cv')].mean()
        cv_data.append(row)
    
    cv_df = pd.DataFrame(cv_data)
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        for model in models:
            model_cv = variability.loc[model][(metric, 'cv')].dropna()
            axes[idx].hist(model_cv, bins=20, alpha=0.5, label=model)
        axes[idx].set_title(f'{metric.upper()} CV')
        axes[idx].set_xlabel("CV")
        axes[idx].legend()
    plt.suptitle("CV Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cv_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: cv_distribution.png")
    return cv_df


def generate_correlation_matrix(df, output_dir):
    print_info("Generazione: Matrice Correlazione")
    cols = ['prompt_char_length', 'ttft', 'itl', 'tpot', 'e2e_latency', 'throughput', 'output_tokens']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax, square=True)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Salvato: correlation_matrix.png")


def generate_report(df, output_dir, cv_df, prompt_corr, output_corr):
    print_info("Generazione: Report Dettagliato")
    
    models = df['model'].unique()
    report_file = f"{output_dir}/analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("         LLM BENCHMARKING - REPORT ANALISI DETTAGLIATO\n")
        f.write(f"         Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(METRICS_DOC)
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("                         PANORAMICA DATASET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Record totali: {len(df)}\n")
        f.write(f"Modelli: {', '.join(models)}\n")
        f.write(f"Prompt unici: {df['prompt_id'].nunique()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("                    STATISTICHE PER MODELLO\n")
        f.write("="*80 + "\n\n")
        
        for model in models:
            model_data = df[df['model'] == model]
            f.write(f"\n{'─'*40}\nMODELLO: {model}\n{'─'*40}\n\n")
            
            for metric in ['ttft', 'itl', 'tpot', 'e2e_latency', 'throughput']:
                if metric not in model_data.columns:
                    continue
                s = summary_stats(model_data[metric].dropna())
                f.write(f"  {metric.upper()}:\n")
                f.write(f"    Media: {s['mean']:.6f}, Std: {s['std']:.6f}, CV: {s['cv']:.4f}\n")
                f.write(f"    IC 95%: [{s['ci_95_low']:.6f}, {s['ci_95_high']:.6f}]\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("                      CORRELAZIONI\n")
        f.write("="*80 + "\n\n")
        
        f.write("Prompt Length vs Metriche:\n")
        for model in models:
            f.write(f"  {model}:\n")
            for metric, r in prompt_corr.get(model, {}).items():
                sig = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.1 else ""
                f.write(f"    {metric}: r = {r:.4f} {sig}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("                      VARIABILITÀ (CV)\n")
        f.write("="*80 + "\n\n")
        f.write(cv_df.to_string(index=False))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("                      CONCLUSIONI\n")
        f.write("="*80 + "\n\n")
        
        summary = df.groupby('model').agg({'ttft': 'mean', 'e2e_latency': 'mean', 'throughput': 'mean'})
        f.write(f"  • Miglior TTFT: {summary['ttft'].idxmin()} ({summary['ttft'].min():.4f}s)\n")
        f.write(f"  • Miglior Throughput: {summary['throughput'].idxmax()} ({summary['throughput'].max():.2f} tok/s)\n")
        f.write(f"  • Miglior E2E: {summary['e2e_latency'].idxmin()} ({summary['e2e_latency'].min():.4f}s)\n")
    
    print_success(f"Salvato: {report_file}")


def main():
    print_header("LLM BENCHMARK ANALYSIS v2")
    print(f"""
{Colors.CYAN}Software Quality Engineering 2025/2026 - Homework 3{Colors.END}
{Colors.CYAN}Authors: Giovanni Altieri, Matteo Patella{Colors.END}
""")
    
    config = configure_analysis_interactive()
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print_header("CARICAMENTO DATI")
    df = load_results(config['input_file'])
    
    if 'error' in df.columns:
        valid_df = df[df['error'].isna()].copy()
        print_info(f"Record validi: {len(valid_df)}/{len(df)}")
    else:
        valid_df = df.copy()
    
    print_header("ESECUZIONE ANALISI")
    prompt_corr = analyze_prompt_length(valid_df, config['output_dir'])
    output_corr = analyze_output_length(valid_df, config['output_dir'])
    analyze_model_comparison(valid_df, config['output_dir'])
    cv_df = analyze_variability(valid_df, config['output_dir'])
    generate_correlation_matrix(valid_df, config['output_dir'])
    
    print_header("GENERAZIONE REPORT")
    generate_report(valid_df, config['output_dir'], cv_df, prompt_corr, output_corr)
    
    print_header("COMPLETATO")
    print(f"""
{Colors.GREEN}Analisi completata!{Colors.END}

{Colors.BOLD}Output:{Colors.END} {config['output_dir']}

{Colors.CYAN}Report dettagliato: {config['output_dir']}/analysis_report.txt{Colors.END}
""")


if __name__ == "__main__":
    main()
