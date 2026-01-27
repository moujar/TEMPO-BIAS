from .runner import PipelineRunner
from ..config.loader import ConfigLoader
from ..datasets.loader import DatasetLoader
from ..prompts.builder import PromptSpec, PromptBuilder
from ..models.openai import OpenAIAdapter
from ..models.hf import HuggingFaceAdapter
from ..models.vllm import VLLMAdapter
from ..metrics.ic import PredictionInconsistency
from ..outputs.writers import OutputWriter
from ..utils.reproducibility import set_seed
import logging
import os

class PipelineController:
    def __init__(self, config_path: str):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load(config_path)
        self.logger = logging.getLogger("PipelineController")

    def analyze(self):
        # 1. Setup
        set_seed(self.config['experiment']['random_seed'])
        
        # 2. Load Resources
        self.logger.info("Loading dataset...")
        dataset = DatasetLoader(
            self.config['dataset']['path'],
            self.config['dataset'].get('required_columns')
        )
        
        self.logger.info("Initializing prompt builder...")
        prompt_spec = PromptSpec(
            instruction=self.config['prompt']['instruction'],
            few_shot_examples=self.config['prompt']['few_shot_examples'],
            label_space=self.config['prompt']['label_space'],
            language=self.config['prompt'].get('language')
        )
        prompt_builder = PromptBuilder(prompt_spec)
        
        self.logger.info("Initializing model...")
        model_provider = self.config['model']['provider']
        
        if model_provider == 'openai':
            model = OpenAIAdapter(
                self.config['model']['model_name'],
                self.config['model']['inference_params'],
                api_key=os.environ.get(self.config['model']['api_key_env'])
            )
        elif model_provider == 'hf':
            model = HuggingFaceAdapter(
                self.config['model']['model_name'],
                self.config['model']['inference_params']
            )
        elif model_provider == 'vllm':
            model = VLLMAdapter(
                self.config['model']['model_name'],
                self.config['model']['inference_params']
            )
        else:
            raise NotImplementedError(f"Provider {model_provider} not yet implemented")

        # 3. Initialize Outputs
        output_writer = OutputWriter(self.config['experiment']['output_dir'])
        
        # 4. Initialize Metrics
        metrics = []
        if 'IC' in self.config['metrics']['enabled']:
            metrics.append(PredictionInconsistency())
            
        # 5. Run Pipeline
        runner = PipelineRunner(
            dataset=dataset,
            prompt_builder=prompt_builder,
            model=model,
            metrics=metrics,
            output_writer=output_writer,
            config=self.config
        )
        
        self.logger.info("Starting execution...")
        runner.run()
        self.logger.info("Execution complete.")
