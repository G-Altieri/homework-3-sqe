#!/usr/bin/env python3
"""
LLM Benchmarking Script v2 - Interactive Version
Software Quality Engineering 2025/2026 - Homework 3

Authors:
- Giovanni Altieri (309006)
- Matteo Patella (308056)

This script benchmarks LLM inference latency using Ollama.
Features:
- Interactive terminal configuration
- Fixed ITL calculation using Ollama internal metrics
- Additional metrics: TPOT, Prompt Processing Time, Generation Time
"""

import requests
import jsonlines
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import sys

try:
    from simple_term_menu import TerminalMenu
    HAS_MENU = True
except ImportError:
    HAS_MENU = False


# ============================================================
# CONFIGURATION DEFAULTS
# ============================================================

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_PROMPTS_FILE = "prompts_group_6.jsonl"

# Available models (will be updated from Ollama)
AVAILABLE_MODELS = []


# ============================================================
# TERMINAL COLORS
# ============================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars per token in English)."""
    return len(text) // 4


def categorize_prompt_length(char_length: int) -> str:
    """Categorize prompt by character length."""
    if char_length < 200:
        return "short"
    elif char_length < 500:
        return "medium"
    elif char_length < 1000:
        return "long"
    else:
        return "very_long"


def get_available_models() -> list:
    """Get list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return [m['name'] for m in response.json().get('models', [])]
    except:
        pass
    return []


def check_ollama_connection() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f} secondi"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minuti"
    else:
        return f"{seconds/3600:.1f} ore"


# ============================================================
# INTERACTIVE CONFIGURATION
# ============================================================

def select_models_interactive(available_models: list) -> list:
    """Interactive model selection using terminal menu or fallback."""
    print_header("SELEZIONE MODELLI")
    
    if not available_models:
        print_error("Nessun modello trovato in Ollama!")
        print_info("Esegui: ollama pull <nome_modello>")
        return []
    
    print("Modelli disponibili in Ollama:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    if HAS_MENU:
        print(f"\n{Colors.CYAN}Usa SPAZIO per selezionare/deselezionare, INVIO per confermare{Colors.END}")
        
        terminal_menu = TerminalMenu(
            available_models,
            multi_select=True,
            show_multi_select_hint=True,
            multi_select_select_on_accept=False,
            multi_select_empty_ok=False,
            title="Seleziona i modelli da benchmarkare:"
        )
        
        selected_indices = terminal_menu.show()
        
        if selected_indices is None:
            return []
        
        if isinstance(selected_indices, int):
            selected_indices = [selected_indices]
        
        return [available_models[i] for i in selected_indices]
    else:
        # Fallback senza menu grafico
        print(f"\n{Colors.CYAN}Inserisci i numeri dei modelli separati da virgola (es: 1,3,5){Colors.END}")
        print(f"{Colors.CYAN}Oppure 'all' per selezionarli tutti:{Colors.END}")
        
        while True:
            choice = input("\n> ").strip().lower()
            
            if choice == 'all':
                return available_models
            
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [available_models[i] for i in indices if 0 <= i < len(available_models)]
                
                if selected:
                    return selected
                else:
                    print_error("Nessun modello valido selezionato. Riprova.")
            except (ValueError, IndexError):
                print_error("Input non valido. Inserisci numeri separati da virgola.")


def get_integer_input(prompt: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
    """Get an integer input with validation."""
    while True:
        try:
            value = input(f"{prompt} [{Colors.CYAN}{default}{Colors.END}]: ").strip()
            
            if value == "":
                return default
            
            value = int(value)
            
            if min_val <= value <= max_val:
                return value
            else:
                print_error(f"Il valore deve essere tra {min_val} e {max_val}")
        except ValueError:
            print_error("Inserisci un numero valido")


def get_file_path(prompt: str, default: str, must_exist: bool = True) -> str:
    """Get a file path with validation."""
    while True:
        value = input(f"{prompt} [{Colors.CYAN}{default}{Colors.END}]: ").strip()
        
        if value == "":
            value = default
        
        if must_exist:
            if os.path.exists(value):
                return value
            else:
                print_error(f"File non trovato: {value}")
        else:
            return value


def get_directory_path(prompt: str, default: str) -> str:
    """Get a directory path."""
    value = input(f"{prompt} [{Colors.CYAN}{default}{Colors.END}]: ").strip()
    
    if value == "":
        return default
    
    return value


def configure_benchmark_interactive() -> dict:
    """Interactive configuration for the benchmark."""
    print_header("CONFIGURAZIONE BENCHMARK LLM")
    
    # Check Ollama
    print("Verifica connessione Ollama...")
    if not check_ollama_connection():
        print_error("Ollama non è in esecuzione!")
        print_info("Avvia Ollama con: ollama serve")
        sys.exit(1)
    print_success("Ollama connesso\n")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print_error("Nessun modello disponibile in Ollama!")
        print_info("Scarica un modello con: ollama pull <nome_modello>")
        print_info("Modelli consigliati: qwen2.5:0.5b, llama3.2, mistral:7b")
        sys.exit(1)
    
    config = {}
    
    # 1. Select models
    config['models'] = select_models_interactive(available_models)
    
    if not config['models']:
        print_error("Nessun modello selezionato!")
        sys.exit(1)
    
    print_success(f"Modelli selezionati: {', '.join(config['models'])}")
    
    # 2. Prompts file
    print_header("CONFIGURAZIONE PROMPT")
    config['prompts_file'] = get_file_path(
        "Percorso file prompts (JSONL)",
        DEFAULT_PROMPTS_FILE,
        must_exist=True
    )
    
    # Count prompts
    with jsonlines.open(config['prompts_file'], 'r') as reader:
        total_prompts = sum(1 for _ in reader)
    print_info(f"Prompt totali nel file: {total_prompts}")
    
    # 3. Number of prompts to use
    config['num_prompts'] = get_integer_input(
        "Quanti prompt vuoi analizzare?",
        default=total_prompts,
        min_val=1,
        max_val=total_prompts
    )
    
    # 4. Number of iterations
    print_header("CONFIGURAZIONE ITERAZIONI")
    print_info("Più iterazioni = risultati più affidabili, ma più tempo")
    print_info("Consigliato: 5-10 iterazioni")
    
    config['iterations'] = get_integer_input(
        "Quante iterazioni per ogni prompt?",
        default=8,
        min_val=1,
        max_val=50
    )
    
    # 5. Max tokens
    config['max_tokens'] = get_integer_input(
        "Numero massimo di token da generare",
        default=256,
        min_val=1,
        max_val=4096
    )
    
    # 6. Output directory
    print_header("CONFIGURAZIONE OUTPUT")
    config['output_dir'] = get_directory_path(
        "Directory di output",
        DEFAULT_OUTPUT_DIR
    )
    
    # Calculate estimated time
    print_header("RIEPILOGO CONFIGURAZIONE")
    
    total_inferences = len(config['models']) * config['num_prompts'] * config['iterations']
    
    # Estimate time based on model sizes
    estimated_time = 0
    for model in config['models']:
        if '0.5b' in model.lower() or '1b' in model.lower():
            estimated_time += config['num_prompts'] * config['iterations'] * 4
        elif '3b' in model.lower() or '2b' in model.lower():
            estimated_time += config['num_prompts'] * config['iterations'] * 6
        elif '7b' in model.lower() or '8b' in model.lower():
            estimated_time += config['num_prompts'] * config['iterations'] * 15
        else:
            estimated_time += config['num_prompts'] * config['iterations'] * 8
    
    print(f"""
{Colors.BOLD}Configurazione:{Colors.END}
  • Modelli: {', '.join(config['models'])}
  • File prompt: {config['prompts_file']}
  • Numero prompt: {config['num_prompts']}
  • Iterazioni: {config['iterations']}
  • Max token: {config['max_tokens']}
  • Directory output: {config['output_dir']}
  
{Colors.BOLD}Stima:{Colors.END}
  • Totale inferenze: {total_inferences}
  • Tempo stimato: ~{format_time(estimated_time)}
""")
    
    # Confirm
    confirm = input(f"\n{Colors.YELLOW}Vuoi procedere con il benchmark? (s/n){Colors.END} [s]: ").strip().lower()
    
    if confirm not in ['', 's', 'si', 'y', 'yes']:
        print_info("Benchmark annullato.")
        sys.exit(0)
    
    return config


# ============================================================
# BENCHMARKING FUNCTIONS
# ============================================================

def generate_with_stream(model: str, prompt: str, options: dict = None) -> dict:
    """
    Execute inference with streaming to capture detailed metrics.
    
    Metrics collected:
    - e2e_latency: Total time from request to last token
    - ttft: Time to first token (from Ollama: load + prompt processing)
    - itl: Inter-token latency (from Ollama: eval_duration / (eval_count - 1))
    - tpot: Time per output token (eval_duration / eval_count)
    - throughput: Tokens per second
    - prompt_processing_time: Time to process input prompt
    - generation_time: Pure token generation time
    """
    start = time.time()
    
    try:
        resp = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options,
            },
            timeout=600
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    first_token_time = None
    token_times = []
    chunks = []
    last_data = None
    
    for line in resp.iter_lines():
        if not line:
            continue
        
        data = json.loads(line.decode("utf-8"))
        last_data = data
        
        token = data.get("response", "")
        if not token:
            continue
        
        now = time.time()
        
        if first_token_time is None:
            first_token_time = now
        
        token_times.append(now)
        chunks.append(token)
    
    end = time.time()
    e2e_latency = end - start
    
    # TTFT measured from HTTP
    ttft_measured = (first_token_time - start) if first_token_time else 0.0
    
    # ITL measured from HTTP (less accurate)
    if len(token_times) > 1:
        inter_token_latencies = [token_times[i+1] - token_times[i] 
                                  for i in range(len(token_times)-1)]
        itl_measured = np.mean(inter_token_latencies)
    else:
        itl_measured = 0.0
    
    output_tokens = len(chunks)
    
    # Extract accurate metrics from Ollama
    ollama_metrics = {}
    itl_ollama = 0.0
    tpot = 0.0
    prompt_processing_time = 0.0
    generation_time = 0.0
    ttft_ollama = 0.0
    
    if last_data:
        total_duration = last_data.get("total_duration", 0) * 1e-9
        load_duration = last_data.get("load_duration", 0) * 1e-9
        prompt_eval_count = last_data.get("prompt_eval_count", 0)
        prompt_eval_duration = last_data.get("prompt_eval_duration", 0) * 1e-9
        eval_count = last_data.get("eval_count", 0)
        eval_duration = last_data.get("eval_duration", 0) * 1e-9
        
        ollama_metrics = {
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
        }
        
        prompt_processing_time = prompt_eval_duration
        generation_time = eval_duration
        
        if eval_count > 0:
            tpot = eval_duration / eval_count
        
        if eval_count > 1:
            itl_ollama = eval_duration / (eval_count - 1)
        elif eval_count == 1:
            itl_ollama = eval_duration
        
        ttft_ollama = load_duration + prompt_eval_duration
    
    # Throughput
    if generation_time > 0:
        throughput = output_tokens / generation_time
    elif e2e_latency > 0:
        throughput = output_tokens / e2e_latency
    else:
        throughput = 0
    
    return {
        "error": None,
        "response": "".join(chunks),
        "e2e_latency": e2e_latency,
        "ttft": ttft_ollama if ttft_ollama > 0 else ttft_measured,
        "itl": itl_ollama,
        "ttft_measured": ttft_measured,
        "itl_measured": itl_measured,
        "tpot": tpot,
        "throughput": throughput,
        "output_tokens": output_tokens,
        "prompt_processing_time": prompt_processing_time,
        "generation_time": generation_time,
        "ollama_metrics": ollama_metrics
    }


def run_benchmark(config: dict) -> pd.DataFrame:
    """Run the complete benchmark."""
    
    # Load prompts
    prompts = []
    with jsonlines.open(config['prompts_file'], 'r') as reader:
        prompts = list(reader)
    
    # Limit prompts if needed
    prompts = prompts[:config['num_prompts']]
    
    options = {
        "num_predict": config['max_tokens'],
        "temperature": 0
    }
    
    all_results = []
    total_iterations = len(config['models']) * len(prompts) * config['iterations']
    
    pbar = tqdm(total=total_iterations, desc="Benchmarking", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for model in config['models']:
        print(f"\n{Colors.BOLD}Benchmarking: {model}{Colors.END}")
        
        # Warm-up
        print_info("Warm-up run...")
        _ = generate_with_stream(model, "Hello, how are you?", options)
        time.sleep(1)
        
        for prompt_data in prompts:
            prompt_id = prompt_data['id']
            prompt_text = prompt_data['prompt']
            
            for iteration in range(config['iterations']):
                pbar.set_description(
                    f"{model} | P{prompt_id} | I{iteration+1}/{config['iterations']}"
                )
                
                result = generate_with_stream(model, prompt_text, options)
                
                record = {
                    "model": model,
                    "prompt_id": prompt_id,
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt_text,
                    "prompt_char_length": len(prompt_text),
                    "prompt_estimated_tokens": estimate_tokens(prompt_text),
                    "prompt_length_category": categorize_prompt_length(len(prompt_text)),
                }
                
                if result.get("error"):
                    record["error"] = result["error"]
                    for metric in ["response", "e2e_latency", "ttft", "itl", 
                                   "ttft_measured", "itl_measured", "tpot",
                                   "throughput", "output_tokens",
                                   "prompt_processing_time", "generation_time"]:
                        record[metric] = None
                else:
                    record["error"] = None
                    record["response"] = result["response"]
                    record["e2e_latency"] = result["e2e_latency"]
                    record["ttft"] = result["ttft"]
                    record["itl"] = result["itl"]
                    record["ttft_measured"] = result["ttft_measured"]
                    record["itl_measured"] = result["itl_measured"]
                    record["tpot"] = result["tpot"]
                    record["throughput"] = result["throughput"]
                    record["output_tokens"] = result["output_tokens"]
                    record["prompt_processing_time"] = result["prompt_processing_time"]
                    record["generation_time"] = result["generation_time"]
                    
                    if result["ollama_metrics"]:
                        for k, v in result["ollama_metrics"].items():
                            record[f"ollama_{k}"] = v
                
                all_results.append(record)
                pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(all_results)


def save_results(df: pd.DataFrame, config: dict):
    """Save benchmark results."""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    csv_file = f"{config['output_dir']}/benchmark_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print_success(f"CSV salvato: {csv_file}")
    
    # Save JSONL
    jsonl_file = f"{config['output_dir']}/benchmark_results_{timestamp}.jsonl"
    with jsonlines.open(jsonl_file, "w") as writer:
        writer.write_all(df.to_dict('records'))
    print_success(f"JSONL salvato: {jsonl_file}")
    
    # Save summary
    summary_file = f"{config['output_dir']}/benchmark_summary_{timestamp}.txt"
    save_summary(df, config, summary_file)
    print_success(f"Summary salvato: {summary_file}")
    
    return csv_file, jsonl_file, summary_file


def save_summary(df: pd.DataFrame, config: dict, filepath: str):
    """Save a summary report."""
    valid_df = df[df['error'].isna()]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LLM BENCHMARKING SUMMARY REPORT v2\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURAZIONE\n")
        f.write("-"*40 + "\n")
        f.write(f"Modelli: {', '.join(config['models'])}\n")
        f.write(f"Prompt analizzati: {config['num_prompts']}\n")
        f.write(f"Iterazioni: {config['iterations']}\n")
        f.write(f"Max token: {config['max_tokens']}\n\n")
        
        f.write("RISULTATI\n")
        f.write("-"*40 + "\n")
        f.write(f"Record totali: {len(df)}\n")
        f.write(f"Record validi: {len(valid_df)}\n")
        f.write(f"Errori: {len(df) - len(valid_df)}\n\n")
        
        if len(valid_df) > 0:
            summary = valid_df.groupby('model').agg({
                'ttft': 'mean',
                'itl': 'mean',
                'tpot': 'mean',
                'e2e_latency': 'mean',
                'throughput': 'mean',
                'output_tokens': 'mean',
            }).round(4)
            
            summary.columns = ['Avg TTFT (s)', 'Avg ITL (s)', 'Avg TPOT (s)', 
                              'Avg E2E (s)', 'Avg Throughput', 'Avg Tokens']
            
            f.write("METRICHE MEDIE PER MODELLO\n")
            f.write("-"*40 + "\n")
            f.write(summary.to_string())
            f.write("\n\n")
            
            # CV
            f.write("COEFFICIENT OF VARIATION (stabilità)\n")
            f.write("-"*40 + "\n")
            for model in valid_df['model'].unique():
                model_data = valid_df[valid_df['model'] == model]
                cv_e2e = model_data['e2e_latency'].std() / model_data['e2e_latency'].mean()
                cv_ttft = model_data['ttft'].std() / model_data['ttft'].mean()
                f.write(f"{model}: E2E CV={cv_e2e:.4f}, TTFT CV={cv_ttft:.4f}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print_header("LLM BENCHMARK TOOL v2")
    print(f"""
{Colors.CYAN}Software Quality Engineering 2025/2026 - Homework 3{Colors.END}
{Colors.CYAN}Authors: Giovanni Altieri, Matteo Patella{Colors.END}

Questo tool esegue benchmark di inferenza su LLM locali usando Ollama.
Misura: TTFT, ITL, TPOT, E2E Latency, Throughput
""")
    
    # Check for required packages
    try:
        import jsonlines
    except ImportError:
        print_error("Package 'jsonlines' non trovato!")
        print_info("Installa con: pip install jsonlines")
        sys.exit(1)
    
    if not HAS_MENU:
        print_warning("Package 'simple-term-menu' non trovato.")
        print_info("Per una migliore esperienza: pip install simple-term-menu")
        print_info("Usando fallback testuale...\n")
    
    # Interactive configuration
    config = configure_benchmark_interactive()
    
    # Run benchmark
    print_header("ESECUZIONE BENCHMARK")
    start_time = datetime.now()
    
    results_df = run_benchmark(config)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results
    print_header("SALVATAGGIO RISULTATI")
    csv_file, jsonl_file, summary_file = save_results(results_df, config)
    
    # Final summary
    print_header("BENCHMARK COMPLETATO")
    print(f"""
{Colors.GREEN}Benchmark completato con successo!{Colors.END}

{Colors.BOLD}Durata totale:{Colors.END} {format_time(duration)}
{Colors.BOLD}File generati:{Colors.END}
  • {csv_file}
  • {jsonl_file}
  • {summary_file}

{Colors.CYAN}Per analizzare i risultati, esegui:{Colors.END}
  python analyze_results_v2.py
""")


if __name__ == "__main__":
    main()
