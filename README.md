# LLM Benchmarking Tool - Setup e Utilizzo

## 📋 Prerequisiti

- **Python**: 3.11.9 o superiore
- **Ollama**: Per eseguire modelli LLM localmente
- **Sistema Operativo**: Linux, macOS o Windows

---

## 🚀 Installazione

### 1. Installare Ollama

#### **Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### **macOS**
```bash
# Scarica il pacchetto .dmg da https://ollama.com/download
# Oppure con Homebrew:
brew install ollama
```

#### **Windows**
```powershell
# Scarica l'installer da https://ollama.com/download
# Esegui OllamaSetup.exe
```

Verifica l'installazione:
```bash
ollama --version
```

Avvia il server Ollama (se non parte automaticamente):
```bash
ollama serve
```

---

### 2. Scaricare i Modelli LLM

Scarica i modelli richiesti per il benchmark:

```bash
# Modello leggero (0.5B parametri) - ~350MB
ollama pull qwen2.5:0.5b

# Modello medio (3B parametri) - ~2GB
ollama pull llama3.2

# Modello grande (7B parametri) - ~4.7GB
ollama pull mistral:7b
```

Verifica i modelli installati:
```bash
ollama list
```

---

### 3. Setup Ambiente Python

#### **Crea l'ambiente virtuale**

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat
```

#### **Installa le dipendenze**

```bash
# Installazione base (da requirements.txt)
pip install -r requirements.txt

# Oppure installazione manuale
pip install requests jsonlines pandas numpy matplotlib seaborn tqdm scipy notebook simple-term-menu
```

**Nota per macOS/Linux con NumPy 2.x issues:**
```bash
pip install pandas "numpy<2.0" notebook requests jsonlines matplotlib seaborn tqdm scipy simple-term-menu
```

---

## 📊 Esecuzione Benchmark

### 1. Preparazione File Prompts

Assicurati di avere il file `prompts_group_6.jsonl` nella directory corrente. Il formato deve essere:

```json
{"id": "prompt_1", "prompt": "Your prompt text here"}
{"id": "prompt_2", "prompt": "Another prompt"}
```

---

### 2. Eseguire il Benchmark

#### **Modalità Interattiva** (consigliata)

```bash
python benchmark_v2.py
```

Il tool ti guiderà attraverso una configurazione interattiva dove potrai:
- ✅ Selezionare i modelli da testare (`qwen2.5:0.5b`, `llama3.2`, `mistral:7b`)
- ✅ Scegliere il numero di prompt da analizzare
- ✅ Impostare il numero di iterazioni (consigliato: 5-10)
- ✅ Configurare il numero massimo di token generati
- ✅ Visualizzare una stima del tempo di esecuzione

**Esempio di configurazione consigliata:**
- **Modelli**: `qwen2.5:0.5b`, `llama3.2`, `mistral:7b`
- **Prompt**: Tutti disponibili nel file
- **Iterazioni**: 8
- **Max tokens**: 256
- **Output dir**: `benchmark_results`

---

### 3. Risultati del Benchmark

I risultati verranno salvati in `benchmark_results/` con timestamp:

```
benchmark_results/
├── benchmark_results_20260121_143052.csv      # Dati grezzi
├── benchmark_results_20260121_143052.jsonl    # Formato JSONL
└── benchmark_summary_20260121_143052.txt      # Riepilogo metriche
```

---

## 📈 Analisi Risultati

### Eseguire l'Analisi Statistica

```bash
python analyze_results_v2.py
```

Il tool di analisi:
- 🔍 Rileva automaticamente gli ultimi file di benchmark
- 📊 Genera grafici comparativi tra modelli
- 📉 Calcola correlazioni e variabilità
- 📝 Crea un report dettagliato con interpretazione metriche

---

### Output dell'Analisi

I risultati dell'analisi vengono salvati in `benchmark_results/results_analyzed/`:

```
benchmark_results/results_analyzed/
├── analysis_report.txt                    # Report completo con guida metriche
├── ttft_vs_prompt_length.png             # TTFT vs lunghezza prompt
├── e2e_vs_output_length.png              # Latenza E2E vs token generati
├── model_comparison_boxplot.png          # Distribuzione metriche (boxplot)
├── model_comparison_violin.png           # Distribuzione metriche (violin)
├── model_comparison_heatmap.png          # Performance normalizzate
├── cv_distribution.png                   # Variabilità dei risultati
└── correlation_matrix.png                # Matrice correlazioni
```

---

## 📊 Metriche Principali

Il benchmark misura:

| Metrica | Descrizione | Formula |
|---------|-------------|---------|
| **TTFT** | Time To First Token | `load_duration + prompt_eval_duration` |
| **ITL** | Inter-Token Latency | `eval_duration / (eval_count - 1)` |
| **TPOT** | Time Per Output Token | `eval_duration / eval_count` |
| **E2E** | End-to-End Latency | Tempo totale richiesta-risposta |
| **Throughput** | Token generati/secondo | `output_tokens / generation_time` |
| **CV** | Coefficient of Variation | Stabilità risultati |

---

## 🛠️ Troubleshooting

### Ollama non risponde
```bash
# Verifica che Ollama sia in esecuzione
curl http://localhost:11434/api/tags

# Riavvia il server
pkill ollama
ollama serve
```

### Modello non trovato
```bash
# Verifica modelli installati
ollama list

# Reinstalla modello mancante
ollama pull qwen2.5:0.5b
```

### Errori di memoria (RAM insufficiente)
```bash
# Usa solo modelli piccoli
ollama pull qwen2.5:0.5b
ollama pull llama3.2

# Riduci il numero di prompt/iterazioni nel benchmark
```

### Problemi con NumPy 2.x
```bash
pip uninstall numpy
pip install "numpy<2.0"
```

---

## 💡 Consigli per Benchmark Ottimali

1. **Prima esecuzione**: Testa con 1-2 modelli leggeri
2. **Iterazioni**: 5-10 per risultati statisticamente significativi
3. **Hardware**: Modelli 7B+ richiedono almeno 8GB RAM
4. **Tempo**: Budget ~15-30 minuti per 3 modelli × 50 prompt × 8 iterazioni

---

## 👥 Autori

- **Giovanni Altieri** (309006)
- **Matteo Patella** (308056)

*Software Quality Engineering 2025/2026 - Homework 3*

---

## 📚 Risorse Aggiuntive

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Modelli disponibili](https://ollama.com/library)