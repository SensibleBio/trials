"""Core module for the Prompt Management System."""

import json
import os
import logging
import threading
import time 
import yaml 
import jinja2
from typing import Dict, List, Optional, Any, Union, Set, Tuple 
from packaging.version import parse as parse_version, InvalidVersion
from enum import Enum, auto

from jsonschema import validate, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LLMProviderType(Enum):
    """Enum for supported LLM providers."""
    OPENAI = auto()
    GEMINI = auto()

def jinja_filter_bullet_list(value: Union[List[Any], Set[Any], Tuple[Any, ...]]) -> str:
    if not isinstance(value, (list, set, tuple)) or not value: return ""
    return "\n".join(f"- {item}" for item in value)

def jinja_filter_format_codes(value: List[Dict[str, Any]]) -> str:
    if not isinstance(value, list):
        logger.warning("format_codes filter expected a list, got %s.", type(value).__name__)
        return "[Invalid input for format_codes]"
    items = []
    for item in value:
        if isinstance(item, dict):
            code = item.get("code", "N/A"); confidence = item.get("confidence", "N/A")
            items.append(f"- {code} (Confidence: {confidence})")
        else:
            logger.warning("format_codes filter found non-dict item in list: %s", str(item)) 
            items.append(f"- [Invalid item: {type(item).__name__}]")
    if not items: return "[No codes to display]"
    return "\n".join(items)

class PromptManager:
    def __init__(self, base_path: str = "prompts", 
                 template_ttl_seconds: Optional[int] = None, 
                 reference_data_ttl_seconds: Optional[int] = None,
                 provider_type: LLMProviderType = LLMProviderType.OPENAI):
        self.base_path = os.path.abspath(base_path)
        self.templates_dir = os.path.join(self.base_path, "templates")
        self.reference_data_dir = os.path.join(self.base_path, "reference_data")
        self.config_dir = os.path.join(self.base_path, "config")
        self.template_ttl = template_ttl_seconds
        self.ref_data_ttl = reference_data_ttl_seconds
        self.provider_type = provider_type
        
        logger.info(f"PromptManager initialized with base_path: {self.base_path}")
        logger.info(f"Templates directory: {self.templates_dir}")
        logger.info(f"Reference data directory: {self.reference_data_dir}")
        logger.info(f"Config directory: {self.config_dir}")
        logger.info(f"Template cache TTL: {'Indefinite' if not self.template_ttl or self.template_ttl <= 0 else f'{self.template_ttl} seconds'}")
        logger.info(f"Reference data cache TTL: {'Indefinite' if not self.ref_data_ttl or self.ref_data_ttl <= 0 else f'{self.ref_data_ttl} seconds'}")
        logger.info(f"LLM Provider Type: {self.provider_type.name}")
        
        self.template_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.reference_data_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_lock = threading.RLock()
        self.loaded_template_schema: Optional[Dict[str, Any]] = None
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            undefined=jinja2.StrictUndefined, autoescape=False,
            trim_blocks=True, lstrip_blocks=True
        )
        self.jinja_env.filters['bullet_list'] = jinja_filter_bullet_list
        self.jinja_env.filters['format_codes'] = jinja_filter_format_codes
        logger.info(f"Jinja2 environment initialized with FileSystemLoader for path: {self.templates_dir}, and custom filters registered.")

    def set_provider_type(self, provider_type: LLMProviderType) -> None:
        """Update the LLM provider type."""
        self.provider_type = provider_type
        logger.info(f"LLM Provider Type updated to: {self.provider_type.name}")

    def _parse_filename_for_group_and_version(self, filename: str) -> Optional[Tuple[str, str]]:
        if filename.endswith(".yaml") and "_v" in filename:
            name_part, version_part = filename[:-5].split("_v", 1)
            if name_part and not name_part.startswith('_') and version_part:
                return name_part, "v" + version_part
        logger.debug(f"Filename '{filename}' does not match expected template pattern.")
        return None

    def _get_version_sort_key(self, version_str_part_no_v: str) -> Any:
        """
        Generates a sort key for a version string (part after 'v', e.g., "1.0", "2.1beta1").
        Uses the 'packaging.version' library for robust semantic version parsing.
        """
        try:
            # packaging.version.parse can handle a wide range of version formats correctly.
            # It returns a Version object which is directly comparable.
            return parse_version(version_str_part_no_v)
        except InvalidVersion:
            logger.warning(
                f"Version string '{version_str_part_no_v}' could not be parsed by 'packaging.version'. "
                f"Using a fallback sort key which may not sort semantically."
            )
            # Fallback for unparseable strings: sort them very early and alphabetically.
            # Using a tuple that starts with a type that sorts before Version objects if direct comparison happens,
            # or rely on python's mixed type sort error if it's not caught (though Version should handle it).
            # A common fallback is to make them sort "first" or "last" in a predictable way.
            # Sorting them "first" and then alphabetically:
            return (float('-inf'), version_str_part_no_v) 
        except Exception as e: # Catch any other unexpected error from parse_version
            logger.error(
                f"Unexpected error parsing version string '{version_str_part_no_v}' with 'packaging.version': {e}. "
                f"Using fallback sort."
            )
            return (float('-inf'), version_str_part_no_v)


    def _validate_template_data(self, template_data: Dict[str, Any], template_filepath: str) -> bool:
        if self.loaded_template_schema is None:
            with self.cache_lock: 
                if self.loaded_template_schema is None:
                    schema_filename = "template_schema.json"; schema_filepath = os.path.join(self.templates_dir, schema_filename)
                    try:
                        with open(schema_filepath, 'r', encoding='utf-8') as f: self.loaded_template_schema = json.load(f)
                        logger.info("Template schema loaded and cached successfully.")
                    except Exception as e: logger.error(f"CRITICAL: Error loading schema {schema_filepath}: {e}. Validation cannot proceed."); return False
        if not self.loaded_template_schema: logger.error("Schema not available. Validation cannot proceed."); return False
        try:
            validate(instance=template_data, schema=self.loaded_template_schema)
            logger.debug(f"Validation SUCCESS for {template_filepath}.")
            return True
        except ValidationError as e: logger.warning(f"Template validation FAILED for {template_filepath}: {e.message} (Path: {list(e.path)})"); return False
        except Exception as e: logger.error(f"Unexpected validation error for {template_filepath}: {e}", exc_info=True); return False

    def _load_template_file(self, template_group_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        filename_to_load = None
        if version:
            # Fix: strip leading 'v' if present
            version_str = version[1:] if version.startswith('v') else version
            filename_to_load = f"{template_group_name}_v{version_str}.yaml"
        else: # Find latest version
            candidate_files = []
            prefix = f"{template_group_name}_v"; suffix = ".yaml"
            try:
                for fname in os.listdir(self.templates_dir):
                    parsed_info = self._parse_filename_for_group_and_version(fname)
                    if parsed_info and parsed_info[0] == template_group_name: candidate_files.append(fname)
            except OSError as e: logger.error(f"Error listing templates in {self.templates_dir}: {e}"); return None
            if not candidate_files: logger.error(f"No versioned files found for group '{template_group_name}' in {self.templates_dir}"); return None
            
            # Sort using _get_version_sort_key, stripping "group_name_v" and ".yaml"
            # The key function now returns Version objects which are directly comparable.
            candidate_files.sort(key=lambda f: self._get_version_sort_key(f[len(template_group_name) + 2 : -len(suffix)]), reverse=True)
            
            if candidate_files: filename_to_load = candidate_files[0]
            logger.info(f"Latest version for '{template_group_name}' resolved to: {filename_to_load}")
        
        if not filename_to_load: logger.error(f"Could not determine filename for {template_group_name} v{version or 'latest'}."); return None
        
        filepath = os.path.join(self.templates_dir, filename_to_load)
        try:
            with open(filepath, 'r', encoding='utf-8') as f: template_content = yaml.safe_load(f)
            if not isinstance(template_content, dict): logger.error(f"Content of {filepath} is not a dictionary."); return None
            if not self._validate_template_data(template_content, filepath): return None 
            logger.info(f"Successfully loaded and validated template: {filepath}")
            return template_content
        except Exception as e: logger.error(f"Error loading or parsing template file {filepath}: {e}", exc_info=True); return None

    def _get_or_load_template(self, template_group_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        cache_key = f"{template_group_name}_v{version}" if version else f"{template_group_name}_#latest#"
        with self.cache_lock:
            if cache_key in self.template_cache:
                data, ts = self.template_cache[cache_key]
                if not (self.template_ttl and self.template_ttl > 0 and (time.monotonic() - ts) > self.template_ttl):
                    logger.info(f"Cache HIT template: {cache_key}"); return data
                logger.info(f"Template cache item '{cache_key}' is stale (age: {time.monotonic() - ts:.2f}s, TTL: {self.template_ttl}s). Will reload.")
            action = "Reloading stale" if cache_key in self.template_cache else "Loading"
            logger.info(f"{action} template for key '{cache_key}' (not in cache or stale). Attempting to load from file...")
            template_data = self._load_template_file(template_group_name, version)
            if template_data:
                self.template_cache[cache_key] = (template_data, time.monotonic()); return template_data
            if cache_key in self.template_cache: del self.template_cache[cache_key]; logger.info(f"Removed stale {cache_key} post failed reload.")
            return None

    def _load_reference_data_file(self, data_filename_no_ext: str) -> Optional[Any]:
        filepath = os.path.join(self.reference_data_dir, f"{data_filename_no_ext}.json")
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            logger.info(f"Loaded ref data: {filepath}"); return data
        except Exception as e: logger.error(f"Error loading ref data {filepath}: {e}"); return None

    def render_template(self, template_group_name: str, version: Optional[str] = None, **context: Any) -> Union[str, Tuple[str, str]]:
        """
        Render a template for the current LLM provider.
        
        Returns:
            For OpenAI: Returns a tuple of (system_prompt, user_prompt)
            For Gemini: Returns a single combined prompt string
        """
        template_data = self._get_or_load_template(template_group_name, version)
        if not template_data:
            error_msg = f"ERROR: Template {template_group_name} v{version or 'latest'} not available."
            return (error_msg, error_msg) if self.provider_type == LLMProviderType.OPENAI else error_msg

        # Get the appropriate template based on provider
        if self.provider_type == LLMProviderType.OPENAI:
            system_template_str = template_data.get("system_template", "")
            user_template_str = template_data.get("user_template", "")
            if not isinstance(user_template_str, str) or not user_template_str:
                error_msg = f"ERROR: user_template invalid/missing for {template_group_name} v{version or 'latest'}."
                return (error_msg, error_msg)
        else:  # Gemini
            user_template_str = template_data.get("user_template", "")
            if not isinstance(user_template_str, str) or not user_template_str:
                error_msg = f"ERROR: user_template invalid/missing for {template_group_name} v{version or 'latest'}."
                return error_msg

        # Prepare rendering context
        rendering_context: Dict[str, Any] = {}
        try:
            if os.path.isdir(self.reference_data_dir):
                for filename in os.listdir(self.reference_data_dir):
                    if filename.endswith(".json"):
                        data_name = os.path.splitext(filename)[0]
                        ref_data_content = self.get_reference_data(data_name)
                        rendering_context[data_name] = ref_data_content
        except Exception as e:
            logger.error(f"Error auto-loading ref data: {e}")
        rendering_context.update(context)

        try:
            if self.provider_type == LLMProviderType.OPENAI:
                # Render both system and user templates for OpenAI
                system_template = self.jinja_env.from_string(system_template_str)
                user_template = self.jinja_env.from_string(user_template_str)
                return (
                    system_template.render(**rendering_context),
                    user_template.render(**rendering_context)
                )
            else:  # Gemini
                # For Gemini, combine system and user templates if both exist
                if "system_template" in template_data:
                    combined_template = f"{template_data['system_template']}\n\n{user_template_str}"
                else:
                    combined_template = user_template_str
                template = self.jinja_env.from_string(combined_template)
                return template.render(**rendering_context)
        except Exception as e:
            error_msg = f"ERROR: Jinja2 render error for {template_group_name} v{version or 'latest'}: {e}"
            return (error_msg, error_msg) if self.provider_type == LLMProviderType.OPENAI else error_msg

    def get_reference_data(self, data_name: str) -> Optional[Any]:
        with self.cache_lock:
            if data_name in self.reference_data_cache:
                data, ts = self.reference_data_cache[data_name]
                if not (self.ref_data_ttl and self.ref_data_ttl > 0 and (time.monotonic() - ts) > self.ref_data_ttl):
                    logger.info(f"Cache HIT ref_data: {data_name}"); return data
                logger.info(f"Reference data cache item '{data_name}' is stale (age: {time.monotonic() - ts:.2f}s, TTL: {self.ref_data_ttl}s). Will reload.")
            action = "Reloading stale" if data_name in self.reference_data_cache else "Loading"
            logger.info(f"{action} reference data for '{data_name}' (not in cache or stale). Attempting to load from file...")
            data_content = self._load_reference_data_file(data_name)
            if data_content is not None:
                self.reference_data_cache[data_name] = (data_content, time.monotonic()); return data_content
            if data_name in self.reference_data_cache: del self.reference_data_cache[data_name]; logger.info(f"Removed stale {data_name} post failed reload.")
            return None

    def list_available_templates(self, include_versions: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        templates_found: Dict[str, List[str]] = {} 
        try:
            if not os.path.isdir(self.templates_dir): logger.warning(f"Templates dir not found: {self.templates_dir}"); return {} if include_versions else []
            for filename in os.listdir(self.templates_dir):
                parsed = self._parse_filename_for_group_and_version(filename)
                if parsed: group, version_v = parsed; templates_found.setdefault(group, []).append(version_v)
            for group in templates_found: templates_found[group].sort(key=lambda v: self._get_version_sort_key(v[1:]))
            return templates_found if include_versions else sorted(list(templates_found.keys()))
        except OSError as e: logger.error(f"Error listing templates: {e}"); return {} if include_versions else []

    def validate_template(self, template_group_name: str, version: Optional[str] = None) -> bool:
        version_for_log = version or 'latest'
        version_to_load = version
        if version_to_load and version_to_load.lower() == 'latest': version_to_load = None 
        if version_to_load and version_to_load.startswith('v'): version_to_load = version_to_load[1:]
        logger.info(f"Explicitly validating template group: '{template_group_name}', version: '{version_for_log}' (resolved to load: '{version_to_load or 'latest'}')...")
        is_valid = self._get_or_load_template(template_group_name, version_to_load) is not None
        logger.info(f"Validation {'successful' if is_valid else 'failed'} for {template_group_name} v{version_for_log}.")
        return is_valid

    def reload_reference_data(self, data_name: Optional[str] = None) -> None:
        with self.cache_lock:
            if data_name:
                if data_name in self.reference_data_cache: del self.reference_data_cache[data_name]; logger.info(f"Cache cleared for ref_data: {data_name}.")
                else: logger.info(f"No cache for ref_data: {data_name}.")
            else: self.reference_data_cache.clear(); logger.info("All ref_data caches cleared.")

if __name__ == '__main__':
    logger.info(f"Module {__name__} loaded.")
