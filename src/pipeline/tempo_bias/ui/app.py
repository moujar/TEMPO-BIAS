
import gradio as gr
import pandas as pd
import os
import yaml
from datetime import datetime
from tempo_bias.pipeline.controller import PipelineController
from tempo_bias.utils.logging import setup_logger

# Setup logger
setup_logger("tempo-bias-ui")

OUTPUT_DIR = "outputs/gradio_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Registry ---
MODEL_REGISTRY = {
    "gpt-3.5-turbo":          {"provider": "openai", "id": "gpt-3.5-turbo"},
    "gpt-4":                  {"provider": "openai", "id": "gpt-4"},
    
    # Llama
    "llama-2-7b":             {"provider": "hf", "id": "meta-llama/Llama-2-7b-hf"},
    "llama-2-7b-chat":        {"provider": "hf", "id": "meta-llama/Llama-2-7b-chat-hf"},
    "llama-3-8b":             {"provider": "hf", "id": "meta-llama/Meta-Llama-3-8B"},
    "llama-3-8b-instruct":    {"provider": "hf", "id": "meta-llama/Meta-Llama-3-8B-Instruct"},
    "llama-3-70b":            {"provider": "hf", "id": "meta-llama/Meta-Llama-3-70B"},
    
    # Mistral
    "mistral-7b":             {"provider": "hf", "id": "mistralai/Mistral-7B-v0.1"},
    "mistral-7b-instruct":    {"provider": "hf", "id": "mistralai/Mistral-7B-Instruct-v0.1"},
    "mixtral-8x7b":           {"provider": "hf", "id": "mistralai/Mixtral-8x7B-v0.1"},
    "mixtral-8x22b":          {"provider": "hf", "id": "mistralai/Mixtral-8x22B-v0.1"},
    
    # Qwen
    "qwen-7b":                {"provider": "hf", "id": "Qwen/Qwen-7B"},
    "qwen-7b-chat":           {"provider": "hf", "id": "Qwen/Qwen-7B-Chat"},
    "qwen-14b":               {"provider": "hf", "id": "Qwen/Qwen-14B"},
    "qwen-72b":               {"provider": "hf", "id": "Qwen/Qwen-72B"},
    
    # vLLM
    "vllm-llama-7b":          {"provider": "vllm", "id": "meta-llama/Llama-2-7b-hf"},
    "vllm-mistral-7b":        {"provider": "vllm", "id": "mistralai/Mistral-7B-v0.1"},
    "vllm-qwen-7b":           {"provider": "vllm", "id": "Qwen/Qwen-7B"},
    
    # Falcon
    "falcon-7b":              {"provider": "hf", "id": "tiiuae/falcon-7b"},
    "falcon-7b-instruct":     {"provider": "hf", "id": "tiiuae/falcon-7b-instruct"},
    "falcon-40b":             {"provider": "hf", "id": "tiiuae/falcon-40b"},
    
    # Other
    "allam-7b":               {"provider": "hf", "id": "ALLaM-LLM/ALLaM-7B-Instruct-preview"}, # Guessing path, might need correction
    "allam-7b-instruct":      {"provider": "hf", "id": "ALLaM-LLM/ALLaM-7B-Instruct-preview"},
    "aya-101":                {"provider": "hf", "id": "CohereForAI/aya-101"},
    "atlas-chat-9b":          {"provider": "hf", "id": "nomic-ai/nomic-embed-text-v1.5"}, # Placeholder, verify ID
    "gemma-7b":               {"provider": "hf", "id": "google/gemma-7b"},
    "gemma-7b-instruct":      {"provider": "hf", "id": "google/gemma-7b-it"},
    "bloom":                  {"provider": "hf", "id": "bigscience/bloom"},
    "bloomz":                 {"provider": "hf", "id": "bigscience/bloomz"},
    "opt-6.7b":               {"provider": "hf", "id": "facebook/opt-6.7b"},
    "opt-13b":                {"provider": "hf", "id": "facebook/opt-13b"},
}

MODEL_CHOICES = list(MODEL_REGISTRY.keys())

def load_dataset_preview(file_obj):
    if file_obj is None:
        return None
    try:
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj.name)
        elif file_obj.name.endswith('.json'):
            df = pd.read_json(file_obj.name)
        else:
            return "Unsupported file format"
        return df.head()
    except Exception as e:
        return str(e)

def run_pipeline(
    dataset_file,
    instruction,
    few_shot_input,
    few_shot_output,
    selected_model_key,
    api_key,
    hf_token,  # New argument
    progress=gr.Progress()
):
    if dataset_file is None:
        raise gr.Error("Please upload a dataset.")
    
    model_info = MODEL_REGISTRY.get(selected_model_key)
    if not model_info:
        raise gr.Error(f"Unknown model: {selected_model_key}")
        
    provider = model_info['provider']
    real_model_name = model_info['id']

    if provider == "openai" and not api_key:
         raise gr.Error("API Key is required for OpenAI models.")
         
    # Handle HF Token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        # Also try to log in programmatically if needed, but env var is usually enough for transformers
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception:
            pass # changes to env var should suffice

    # Note for HF: Running locally means downloading GBs. We should warn user but code will execute.

    # 1. Create a temporary config
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Handle few-shot
    few_shot_examples = []
    if few_shot_input and few_shot_output:
        few_shot_examples.append({"input": few_shot_input, "output": few_shot_output})

    config = {
        "experiment": {
            "name": f"gradio_run_{run_id}",
            "description": "Run triggered via Gradio UI",
            "output_dir": run_dir,
            "random_seed": 42
        },
        "dataset": {
            "path": dataset_file.name,
            "format": "csv" if dataset_file.name.endswith(".csv") else "json",
            "required_columns": ["sentence_id", "template", "entity"]
        },
        "prompt": {
            "instruction": instruction,
            "few_shot_examples": few_shot_examples,
            "label_space": ["Positive", "Negative", "Neutral"],
            "language": "en"
        },
        "model": {
            "provider": provider,
            "model_name": real_model_name,
            "api_key_env": "OPENAI_API_KEY",
            "inference_params": {
                "temperature": 0,
                "max_tokens": 100
            }
        },
        "metrics": {
            "enabled": ["IC"]
        },
        "output": {
            "save_raw": True,
            "save_metrics": True
        }
    }
    
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # 2. Run Pipeline
    try:
        progress(0, desc="Initializing Pipeline...")
        controller = PipelineController(config_path)
        
        progress(0.2, desc=f"Running Analysis with {selected_model_key}...")
        controller.analyze()
        
        progress(1.0, desc="Completed!")
        
        # 3. Load Results
        responses_path = os.path.join(run_dir, "responses.csv")
        metrics_path = os.path.join(run_dir, "metrics.csv")
        
        responses_df = pd.read_csv(responses_path) if os.path.exists(responses_path) else pd.DataFrame()
        metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
        
        return responses_df, metrics_df, f"Run {run_id} completed successfully."
        
    except Exception as e:
        raise gr.Error(f"Pipeline Failed: {str(e)}")


# --- UI Layout ---
with gr.Blocks(title="TEMPO-BIAS Pipeline") as app:
    gr.Markdown("# TEMPO-BIAS: Political Bias Analysis Pipeline")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            
            with gr.Accordion("Credentials", open=True):
                # API Key
                api_key = gr.Textbox(
                    label="OpenAI API Key (Optional)", 
                    type="password", 
                    placeholder="sk-...",
                    info="Required for GPT models."
                )
                # HF Token
                hf_token = gr.Textbox(
                    label="Hugging Face Token (Optional)", 
                    type="password", 
                    placeholder="hf_...",
                    info="Required for gated models (Llama, Mistral, etc)."
                )
            
            # Dataset
            dataset_file = gr.File(
                label="Upload Dataset (CSV/JSON)", 
                file_types=[".csv", ".json"]
            )
            dataset_preview = gr.Dataframe(label="Dataset Preview", interactive=False)
            
            dataset_file.change(load_dataset_preview, inputs=dataset_file, outputs=dataset_preview)
            
            # Prompt Settings
            instruction = gr.Textbox(
                label="Instruction", 
                value="Classify the sentiment of the sentence regarding the entity.",
                lines=2
            )
            
            with gr.Accordion("Few-Shot Example (Optional)", open=False):
                with gr.Row():
                    fs_input = gr.Textbox(label="Example Input", placeholder="The sun is shining.")
                    fs_output = gr.Textbox(label="Example Output", placeholder="Positive")
            
            # Model Settings
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES, 
                value="gpt-3.5-turbo", 
                label="Select Model", 
                info="GPT models use OpenAI API. others run LOCALLY (requires download)."
            )
            
            run_btn = gr.Button("Run Analysis", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Results")
            
            status_msg = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("Metrics"):
                    metrics_df = gr.Dataframe(label="Aggregated Metrics")
                with gr.TabItem("Raw Responses"):
                    responses_df = gr.Dataframe(label="Detailed Responses")

    run_btn.click(
        run_pipeline,
        inputs=[
            dataset_file, instruction, fs_input, fs_output, 
            model_name, api_key, hf_token
        ],
        outputs=[responses_df, metrics_df, status_msg]
    )

if __name__ == "__main__":
    app.launch()
