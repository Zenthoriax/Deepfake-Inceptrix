import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, config_path=None):
        self.models = {}
        self.chains = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load chains based on config
            model_fallbacks = config.get('model_fallbacks', {})
            chains_config = model_fallbacks.get('chains', {})
            
            for task, chain in chains_config.items():
                self.chains[task] = {
                    'primary': chain.get('primary'),
                    'secondary': chain.get('secondary'),
                    'emergency': chain.get('emergency')
                }
            logger.info(f"Loaded {len(self.chains)} model fallback chains from config.")
        except Exception as e:
            logger.error(f"Failed to load fallback config: {e}")

    def register_model(self, name, model_instance):
        self.models[name] = model_instance
        
    def populate_from_stubs(self, stubs_dict):
        """Helper to mass-register models from the stubs dictionary."""
        for name, instance in stubs_dict.items():
            self.register_model(name, instance)

    def get_chain(self, task_name):
        chain_info = self.chains.get(task_name)
        if not chain_info:
            logger.warning(f"No chain found for task {task_name}")
            return None, None, None
            
        return (
            self.models.get(chain_info['primary']),
            self.models.get(chain_info['secondary']),
            self.models.get(chain_info['emergency'])
        )

# Global registry instance
registry = ModelRegistry()

