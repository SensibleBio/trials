import unittest
from unittest.mock import patch, mock_open, MagicMock, ANY
import os
import shutil 
import yaml 
import json 
from jsonschema import ValidationError 
import jinja2 
import threading 
import time 
import re 
from packaging.version import Version, InvalidVersion # For type checking

from prompt_manager import PromptManager, jinja_filter_bullet_list, jinja_filter_format_codes

class TestPromptManager(unittest.TestCase):

    def setUp(self):
        self.base_test_dir_name = "temp_test_prompts_version_sort_focused_v2" 
        os.makedirs(self.base_test_dir_name, exist_ok=True)
        self.manager = PromptManager(base_path=self.base_test_dir_name) 
        self.manager.templates_dir = os.path.join(self.manager.base_path, "templates")
        self.manager.reference_data_dir = os.path.join(self.manager.base_path, "reference_data")
        self.manager.config_dir = os.path.join(self.manager.base_path, "config")
        os.makedirs(self.manager.templates_dir, exist_ok=True)
        os.makedirs(self.manager.reference_data_dir, exist_ok=True)
        os.makedirs(self.manager.config_dir, exist_ok=True)
        
        self.mock_logger_patch = patch('prompt_manager.logger')
        self.mock_logger = self.mock_logger_patch.start()
        
        self.addCleanup(shutil.rmtree, self.base_test_dir_name)
        self.addCleanup(self.mock_logger_patch.stop)
        
        self.manager.template_cache.clear()
        self.manager.reference_data_cache.clear()
        self.manager.loaded_template_schema = None

        self.dummy_schema_path = os.path.join(self.manager.config_dir, "template_schema.json")
        with open(self.dummy_schema_path, 'w') as f:
            json.dump({
                "type": "object", 
                "properties": {"metadata": {"type": "object"}, "user_template": {"type": "string"}},
                "required": ["metadata", "user_template"]
            }, f)


    def test_get_version_sort_key(self):
        valid_versions = ["1.0", "2.0beta1", "1.0.1", "0.9", "10", "1.alpha", "1.0.0-alpha.1"]
        for v_str in valid_versions:
            key = self.manager._get_version_sort_key(v_str)
            self.assertIsInstance(key, Version, f"Expected Version object for '{v_str}', got {type(key)}")

        invalid_version_str = "totally_invalid_version"
        fallback_key = self.manager._get_version_sort_key(invalid_version_str)
        self.assertEqual(fallback_key, (float('-inf'), invalid_version_str))
        self.mock_logger.warning.assert_any_call(
            f"Version string '{invalid_version_str}' could not be parsed by 'packaging.version'. "
            f"Using a fallback sort key which may not sort semantically."
        )
        self.mock_logger.reset_mock() 

        with patch('prompt_manager.parse_version', side_effect=ValueError("Unexpected parsing error")):
            fallback_key_unexpected = self.manager._get_version_sort_key("problem_version")
            self.assertEqual(fallback_key_unexpected, (float('-inf'), "problem_version"))
            self.mock_logger.error.assert_any_call(
                f"Unexpected error parsing version string 'problem_version' with 'packaging.version': Unexpected parsing error. "
                f"Using fallback sort."
            )

        versions_to_test_sort = ["1.0", "10.0", "2.0beta1", "1.0alpha", "0.9", "1.0.1", "1.0.0rc1"] 
        # Corrected expected order to reflect canonical version string for "1.0.0rc1"
        expected_sorted_versions = ["0.9", "1.0alpha", "1.0.0rc1", "1.0", "1.0.1", "2.0beta1", "10.0"]
        
        actual_sorted_versions = sorted(versions_to_test_sort, key=self.manager._get_version_sort_key)
        self.assertEqual(actual_sorted_versions, expected_sorted_versions)

    @patch('prompt_manager.os.listdir')
    @patch('prompt_manager.os.path.isdir')
    def test_list_available_templates_with_versions(self, mock_isdir, mock_listdir):
        mock_isdir.return_value = True
        mock_listdir.return_value = [
            "group1_v1.0.yaml", "group2_v1.yaml", "group1_v2.0.yaml", 
            "group1_v0.9.yaml", "group2_v1.1beta1.yaml", "group1_v1.0alpha2.yaml", 
            "group3_v10.0.yaml", "group3_v2.0.yaml", "_ignored_v1.yaml",
            "group1_v1.0rc1.yaml" 
        ]
        expected = {
            "group1": ["v0.9", "v1.0alpha2", "v1.0rc1", "v1.0", "v2.0"], # Adjusted order
            "group2": ["v1", "v1.1beta1"], 
            "group3": ["v2.0", "v10.0"] 
        }
        result = self.manager.list_available_templates(include_versions=True)
        self.assertEqual(result.get("group1"), expected["group1"])
        self.assertEqual(result.get("group2"), expected["group2"])
        self.assertEqual(result.get("group3"), expected["group3"])
        self.assertEqual(len(result), 3, "Should only include valid groups")

    def test_dummy_always_passes(self): # To ensure test file is runnable
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
