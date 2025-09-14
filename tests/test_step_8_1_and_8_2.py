#!/usr/bin/env python3
"""
Comprehensive test suite for Steps 8.1 and 8.2 of the experiment parameterization system.

Tests:
- Step 8.1: Enhanced experiment hyperparameter logging
- Step 8.2: Complete YAML schema structure and loading
"""

import asyncio
import sys
import json
import yaml
from pathlib import Path
import tempfile
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment
from negotiation.llm_agents import LLMConfig, ModelType, SimulatedAgent


class TestStep8Implementation:
    """Test suite for Steps 8.1 and 8.2."""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.failures = []
    
    async def run_test(self, test_name, test_func):
        """Run a single test and track results."""
        self.total_tests += 1
        print(f"\n🧪 Running {test_name}...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            print(f"✅ {test_name}: PASSED")
            self.passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            self.failures.append((test_name, str(e)))
    
    async def test_step_8_1_enhanced_config_creation(self):
        """Test Step 8.1: Enhanced configuration creation."""
        experiment = O3VsHaikuExperiment()
        
        # Create test agents
        agents = [
            SimulatedAgent("test_o3", LLMConfig(
                model_type=ModelType.O3,
                temperature=0.8,
                max_tokens=2000,
                system_prompt="O3 test prompt",
                custom_parameters={"reasoning_steps": True}
            )),
            SimulatedAgent("test_haiku", LLMConfig(
                model_type=ModelType.CLAUDE_3_HAIKU,
                temperature=0.7,
                max_tokens=1500
            ))
        ]
        
        # Test preferences and items
        preferences = {
            "test_o3": [8.5, 2.3, 6.7, 1.9, 9.1],
            "test_haiku": [3.2, 7.8, 4.5, 8.9, 2.1]
        }
        items = ["Item1", "Item2", "Item3", "Item4", "Item5"]
        
        base_config = {
            "m_items": 5,
            "n_agents": 2,
            "t_rounds": 4,
            "gamma_discount": 0.9,
            "competition_level": 0.85,
            "known_to_all": False,
            "random_seed": 12345
        }
        
        import time
        start_time = time.time()
        
        # Test enhanced config creation
        enhanced_config = experiment._create_enhanced_config(
            base_config, agents, preferences, items, start_time, "test_exp_001"
        )
        
        # Verify all required sections exist
        required_sections = [
            "experiment_metadata",
            "agent_configurations",
            "preference_configuration", 
            "item_configuration",
            "proposal_order_configuration",
            "negotiation_flow_configuration"
        ]
        
        for section in required_sections:
            assert section in enhanced_config, f"Missing section: {section}"
        
        # Verify agent configurations capture hyperparameters
        assert len(enhanced_config["agent_configurations"]) == 2
        o3_config = enhanced_config["agent_configurations"]["test_o3"]
        assert o3_config["model_type"] == "o3"
        assert o3_config["temperature"] == 0.8
        assert o3_config["max_tokens"] == 2000
        assert o3_config["system_prompt"] == "O3 test prompt"
        assert o3_config["custom_parameters"]["reasoning_steps"] is True
        
        # Verify preference configuration
        pref_config = enhanced_config["preference_configuration"]
        assert pref_config["preference_type"] == "vector"
        assert len(pref_config["preferences_by_agent"]) == 2
        assert pref_config["preferences_by_agent"]["test_o3"]["preference_vector"] == [8.5, 2.3, 6.7, 1.9, 9.1]
        
        # Verify JSON serialization works
        json_str = json.dumps(enhanced_config, indent=2, default=str)
        assert len(json_str) > 1000, "Enhanced config JSON should be substantial"
        
        print(f"    📊 Enhanced config has {len(enhanced_config)} sections")
        print(f"    📄 JSON size: {len(json_str)} characters")
    
    def test_step_8_2_yaml_schema_structure(self):
        """Test Step 8.2: YAML schema structure and loading."""
        schema_path = Path("experiments/configs/model_config_schema.yaml")
        assert schema_path.exists(), f"Schema file not found: {schema_path}"
        
        # Test YAML loading
        with open(schema_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict), "Config should be a dictionary"
        
        # Verify all required sections are present
        expected_sections = [
            "config_name", "description", "version",
            "providers", "available_models", "agents",
            "environment", "preferences", "negotiation_flow",
            "proposal_order_analysis", "analysis", 
            "experiment_metadata", "cluster_config", "validation_rules"
        ]
        
        for section in expected_sections:
            assert section in config, f"Missing YAML section: {section}"
        
        print(f"    📋 Schema has {len(config)} main sections")
        print(f"    📊 Available models: {len(config.get('available_models', {}))}")
        
    def test_step_8_2_environment_section(self):
        """Test Step 8.2: Environment configuration section."""
        schema_path = Path("experiments/configs/model_config_schema.yaml")
        
        with open(schema_path, 'r') as f:
            config = yaml.safe_load(f)
        
        env_config = config["environment"]
        
        # Verify core negotiation parameters are present
        core_params = ["num_agents", "num_items", "max_rounds", "discount_factor"]
        for param in core_params:
            assert param in env_config, f"Missing environment parameter: {param}"
        
        # Verify validation rules are present
        validation_sections = ["agent_validation", "item_validation", "round_validation", "discount_validation"]
        for validation in validation_sections:
            assert validation in env_config, f"Missing validation section: {validation}"
        
        # Verify reasonable default values
        assert 2 <= env_config["num_agents"] <= 6, "num_agents should be reasonable"
        assert env_config["num_items"] >= 2, "num_items should be >= 2"
        assert env_config["max_rounds"] >= 1, "max_rounds should be >= 1"
        assert 0.1 <= env_config["discount_factor"] <= 1.0, "discount_factor should be in valid range"
        
        print(f"    🔧 Environment parameters: {len(env_config)} total")
        print(f"    ✅ Core parameters (n,m,t,γ): All present")
        
    def test_step_8_2_preferences_section(self):
        """Test Step 8.2: Preferences configuration section.""" 
        schema_path = Path("experiments/configs/model_config_schema.yaml")
        
        with open(schema_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pref_config = config["preferences"]
        
        # Verify preference type configuration
        assert "preference_type" in pref_config
        assert pref_config["preference_type"] in ["vector", "matrix"]
        
        # Verify vector preferences section
        assert "vector_preferences" in pref_config
        vector_prefs = pref_config["vector_preferences"]
        assert "value_range" in vector_prefs
        assert "distribution" in vector_prefs
        assert "similarity_target" in vector_prefs
        
        # Verify matrix preferences placeholder
        assert "matrix_preferences" in pref_config
        
        # Verify generation parameters
        assert "generation" in pref_config
        gen_params = pref_config["generation"]
        assert "max_attempts" in gen_params
        assert "normalize_preferences" in gen_params
        
        print(f"    🎯 Preference configuration: Complete")
        print(f"    📊 Vector preferences: Configured")
        print(f"    🔮 Matrix preferences: Placeholder ready")
        
    def test_step_8_2_analysis_section(self):
        """Test Step 8.2: Analysis configuration section."""
        schema_path = Path("experiments/configs/model_config_schema.yaml")
        
        with open(schema_path, 'r') as f:
            config = yaml.safe_load(f)
        
        analysis_config = config["analysis"]
        
        # Verify main analysis categories
        analysis_categories = [
            "strategic_behavior_detection",
            "conversation_analysis", 
            "outcome_analysis",
            "statistical_analysis",
            "performance_metrics"
        ]
        
        for category in analysis_categories:
            assert category in analysis_config, f"Missing analysis category: {category}"
        
        # Verify strategic behavior detection parameters
        strategic = analysis_config["strategic_behavior_detection"]
        strategic_metrics = ["detect_manipulation", "detect_anger_expressions", "detect_gaslighting"]
        for metric in strategic_metrics:
            assert metric in strategic, f"Missing strategic metric: {metric}"
        
        # Verify statistical analysis parameters  
        statistical = analysis_config["statistical_analysis"]
        assert "significance_testing" in statistical
        assert "confidence_intervals" in statistical
        assert "multiple_comparison_correction" in statistical
        
        print(f"    📈 Analysis categories: {len(analysis_categories)} configured")
        print(f"    🎯 Strategic behavior detection: Configured")
        print(f"    📊 Statistical analysis: Configured")
    
    async def test_integration_enhanced_config_in_experiment(self):
        """Test Step 8.1: Integration with actual experiment (simulated)."""
        # Create a temporary results directory
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = O3VsHaikuExperiment(results_dir=temp_dir)
            
            # Run a quick simulated experiment
            result = await experiment.run_single_experiment(
                experiment_config={"random_seed": 42}
            )
            
            # Verify enhanced config is present in results
            config = result.config
            
            # Check for enhanced sections
            enhanced_sections = [
                "experiment_metadata",
                "agent_configurations",
                "preference_configuration"
            ]
            
            for section in enhanced_sections:
                assert section in config, f"Enhanced config missing {section} in actual experiment"
            
            # Verify experiment metadata
            metadata = config["experiment_metadata"]
            assert "experiment_id" in metadata
            assert "start_time" in metadata
            assert "duration_seconds" in metadata
            assert metadata["duration_seconds"] >= 0
            
            # Verify agent configurations
            agent_configs = config["agent_configurations"]
            assert len(agent_configs) == 3  # o3_agent, haiku_agent_1, haiku_agent_2
            
            for agent_id, agent_config in agent_configs.items():
                assert "model_type" in agent_config
                assert "temperature" in agent_config
                assert "max_tokens" in agent_config
            
            print(f"    🧪 Experiment ran successfully with enhanced config")
            print(f"    📊 Config sections: {len(config)}")
            print(f"    🤖 Agent configurations: {len(agent_configs)}")
    
    def test_yaml_config_validation(self):
        """Test that YAML config has proper validation rules."""
        schema_path = Path("experiments/configs/model_config_schema.yaml")
        
        with open(schema_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validation_rules = config["validation_rules"]
        
        # Verify key validation rules exist
        expected_rules = ["max_o3_agents", "min_agents", "max_agents", "temperature_range"]
        for rule in expected_rules:
            assert rule in validation_rules, f"Missing validation rule: {rule}"
        
        # Verify reasonable rule values
        assert validation_rules["max_o3_agents"] >= 1
        assert validation_rules["min_agents"] >= 2
        assert validation_rules["max_agents"] <= 10
        assert len(validation_rules["temperature_range"]) == 2
        assert validation_rules["temperature_range"][0] < validation_rules["temperature_range"][1]
        
        print(f"    ✅ Validation rules: {len(validation_rules)} configured")
        print(f"    🎯 Temperature range: {validation_rules['temperature_range']}")
        print(f"    🤖 Agent limits: {validation_rules['min_agents']}-{validation_rules['max_agents']}")
    
    async def run_all_tests(self):
        """Run all tests for Steps 8.1 and 8.2."""
        print("🚀 Testing Steps 8.1 and 8.2 Implementation")
        print("=" * 60)
        
        # Step 8.1 Tests - Enhanced Configuration
        print("\n📋 STEP 8.1: Enhanced Experiment Hyperparameter Logging")
        await self.run_test("Enhanced Config Creation", self.test_step_8_1_enhanced_config_creation)
        await self.run_test("Integration with Experiment", self.test_integration_enhanced_config_in_experiment)
        
        # Step 8.2 Tests - YAML Schema  
        print("\n📋 STEP 8.2: Complete YAML Schema Structure")
        await self.run_test("YAML Schema Loading", self.test_step_8_2_yaml_schema_structure)
        await self.run_test("Environment Section", self.test_step_8_2_environment_section)
        await self.run_test("Preferences Section", self.test_step_8_2_preferences_section)
        await self.run_test("Analysis Section", self.test_step_8_2_analysis_section)
        await self.run_test("Validation Rules", self.test_yaml_config_validation)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 TEST RESULTS: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.failures:
            print(f"\n❌ FAILURES ({len(self.failures)}):")
            for test_name, error in self.failures:
                print(f"  • {test_name}: {error}")
        else:
            print("🎉 All tests PASSED!")
        
        return len(self.failures) == 0


async def main():
    """Main test execution."""
    tester = TestStep8Implementation()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ Steps 8.1 and 8.2 are working correctly!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)