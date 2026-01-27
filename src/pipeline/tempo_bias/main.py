import argparse
import sys
import os
from tempo_bias.pipeline.controller import PipelineController
from tempo_bias.utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description="TEMPO-BIAS Pipeline Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    
    args = parser.parse_args()
    
    # Setup global logger
    setup_logger("tempo-bias")
    
    try:
        controller = PipelineController(args.config)
        controller.analyze()
    except Exception as e:
        print(f"Error during execution: {e}")
        # In debug mode one might want to raise
        # raise e
        sys.exit(1)

if __name__ == "__main__":
    main()
