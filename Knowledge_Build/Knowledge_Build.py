#!/usr/bin/env python3
"""
Java/C++/JavaScript Code Knowledge Base Builder
- Based on Tree-sitter syntax parsing
- Unified graph structure
- Supports Java/C++/JavaScript syntax features
- Includes: AST analysis, call graphs, cross-function data flow graphs
- Includes: LLM function summarization
- Includes: Embedded vector database
"""
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Dict, Set, Optional, Any, Tuple
import numpy as np
import faiss
from tree_sitter import Parser, Node, Language
import requests
from tqdm import tqdm
import logging
from enum import Enum
from pydantic import BaseModel, Field
import networkx as nx
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("java_cpp_js_rag_builder_enhanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Configuration Constants ===
API_URL = ""
API_KEY = ""  # Replace with your API key

# === Language Configuration ===
LANGUAGE_CONFIGS = {
    'cpp': {
        'extensions': {'.c', '.cpp', '.h', '.hpp', '.tcc'},
        'comment_patterns': {
            'single_line': '//',
            'multi_line_start': '/*',
            'multi_line_end': '*/'
        }
    },
    'java': {
        'extensions': {'.java'},
        'comment_patterns': {
            'single_line': '//',
            'multi_line_start': '/*',
            'multi_line_end': '*/',
            'javadoc': True
        }
    },
    'javascript': {
        'extensions': {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'},
        'comment_patterns': {
            'single_line': '//',
            'multi_line_start': '/*',
            'multi_line_end': '*/',
            'javadoc': False
        }
    }
}

# === Global State ===
FUNCTION_CALL_GRAPH = defaultdict(list)
FILE_DEPENDENCIES = defaultdict(list)
CALLED_BY_GRAPH = defaultdict(list)
CLASS_METHOD_GRAPH = defaultdict(list)
IMPORT_GRAPH = defaultdict(list)
DATA_FLOW_GRAPH = defaultdict(list)
FUNCTION_NODE_MAP = {}
CALL_RELATIONS = []
IGNORE_DIRS = {'build', 'cmake-build', '.git', 'vendor', 'lib', 'external', 'Debug',
               '__pycache__', '.pytest_cache', 'target', '.idea', '.vscode', 'node_modules'}

# Added: Cross-function data flow tracking data structures
GLOBAL_VARIABLES = {}  # Global variable mapping {variable_name: {node_id, file, type, is_mutated}}
CLASS_FIELDS = defaultdict(dict)  # Class field mapping {class_name: {field_name: {node_id, type, modifiers}}}
PARAMETER_FLOW = []  # Parameter passing flow [(source_param_id, target_param_id, call_edge_id)]
RETURN_FLOW = []  # Return value flow [(source_return_id, target_var_id, call_edge_id)]
FIELD_ACCESS_FLOW = []  # Field access flow [(method_id, field_id, access_type, line)]
CROSS_FILE_DATA_FLOW = []  # Cross-file data flow [(source_id, target_id, flow_type, file_pair)]


# === Data Type Definitions ===
class NodeType(str, Enum):
    FUNCTION = "Function"
    METHOD = "Method"
    CLASS = "Class"
    INTERFACE = "Interface"
    MODULE = "Module"
    FILE = "File"
    VARIABLE = "Variable"
    PACKAGE = "Package"
    PARAMETER = "Parameter"
    RETURN_VALUE = "ReturnValue"
    FIELD = "Field"
    LITERAL = "Literal"
    EXPRESSION = "Expression"
    CONSTANT = "Constant"
    GLOBAL_VAR = "GlobalVariable"


class EdgeType(str, Enum):
    CALLS = "CALLS"
    CONTAINS = "CONTAINS"
    EXTENDS = "EXTENDS"
    IMPLEMENTS = "IMPLEMENTS"
    IMPORT = "IMPORT"
    INHERITS = "INHERITS"
    DECORATES = "DECORATES"
    OVERRIDES = "OVERRIDES"
    HAS_PARAMETER = "HAS_PARAMETER"
    RETURNS = "RETURNS"
    DEFINES = "DEFINES"
    ASSIGNS_TO = "ASSIGNS_TO"
    READS = "READS"
    WRITES = "WRITES"
    PASSES_TO = "PASSES_TO"  # Parameter passes to callee
    RETURNS_TO = "RETURNS_TO"  # Return value passes back to caller
    DEREFERENCES = "DEREFERENCES"
    ALIAS = "ALIAS"
    USES = "USES"
    MODIFIES = "MODIFIES"
    DATA_DEPENDENCY = "DATA_DEPENDENCY"  # Data dependency
    GLOBAL_ACCESS = "GLOBAL_ACCESS"  # Global variable access
    FIELD_ACCESS = "FIELD_ACCESS"  # Field access


class GraphNode(BaseModel):
    id: str
    type: NodeType
    name: str
    language: str
    signature: Optional[str] = None
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    summary: Optional[str] = None
    logic_flow: Optional[str] = None
    data_flow_info: Optional[str] = None
    embedding: Optional[List[float]] = None
    raw_attributes: Dict[str, Any] = Field(default_factory=dict)
    data_type: Optional[str] = None
    scope: Optional[str] = None
    is_mutated: Optional[bool] = None
    initial_value: Optional[str] = None
    is_global: Optional[bool] = None
    parent_class: Optional[str] = None
    modifiers: List[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    type: EdgeType
    language: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    data_type: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    call_context: Optional[str] = None


@dataclass
class ParameterInfo:
    """Parameter information"""
    name: str
    data_type: str
    position: int
    default_value: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class FunctionSignature:
    """Function signature information"""
    name: str
    return_type: str
    parameters: List[ParameterInfo]
    modifiers: List[str] = field(default_factory=list)


# === Tree-sitter language loading ===
def load_language_parsers():
    """Dynamically load Tree-sitter parsers for Java, C++ and JavaScript"""
    from tree_sitter import Language, Parser
    parsers = {}

    try:
        import tree_sitter_c
        import tree_sitter_cpp
        CPP_LANGUAGE = Language(tree_sitter_cpp.language())
        parser = Parser()
        parser.language = CPP_LANGUAGE
        parsers['cpp'] = parser
    except ImportError as e:
        logger.warning(f"Failed to load C/C++ language support: {e}")

    try:
        import tree_sitter_java
        JAVA_LANGUAGE = Language(tree_sitter_java.language())
        parser = Parser()
        parser.language = JAVA_LANGUAGE
        parsers['java'] = parser
    except ImportError as e:
        logger.warning(f"Failed to load Java language support: {e}")

    # Add: JavaScript parser
    try:
        import tree_sitter_javascript
        JAVASCRIPT_LANGUAGE = Language(tree_sitter_javascript.language())
        parser = Parser()
        parser.language = JAVASCRIPT_LANGUAGE
        parsers['javascript'] = parser
        logger.info("✓ tree-sitter-javascript loaded")
    except ImportError as e:
        logger.warning(f"Failed to load JavaScript language support: {e}")
        logger.warning("Please run: pip install tree-sitter-javascript")

    return parsers


# Load all parsers
LANGUAGE_PARSERS = load_language_parsers()


# === Language detection and utility functions ===
def detect_language(file_path: str) -> Optional[str]:
    """Detect language based on file extension"""
    ext = Path(file_path).suffix.lower()
    for lang, config in LANGUAGE_CONFIGS.items():
        if ext in config['extensions']:
            return lang
    return None


def safe_decode(b: bytes, file_path: str) -> str:
    """Safely decode byte stream"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            return b.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"Cannot decode file {file_path}, using replacement mode")
    return b.decode('utf-8', errors='replace')


def extract_comment_before(node, source_lines: List[bytes], language: str) -> str:
    """Extract comments"""
    start_line = node.start_point[0]
    comment_lines = []

    lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['cpp'])
    single_line = lang_config['comment_patterns'].get('single_line')
    multi_start = lang_config['comment_patterns'].get('multi_line_start')
    multi_end = lang_config['comment_patterns'].get('multi_line_end')

    # Search backwards for comments
    in_multiline = False
    for i in range(start_line - 1, max(0, start_line - 20), -1):
        line = safe_decode(source_lines[i], "").strip()

        if multi_end and line.endswith(multi_end):
            in_multiline = True
            comment_lines.append(line)
            continue

        if multi_start and line.startswith(multi_start):
            in_multiline = False
            comment_lines.append(line)
            break

        if in_multiline:
            comment_lines.append(line)
            continue

        if single_line and line.startswith(single_line):
            comment_lines.append(line)
        elif multi_start and multi_start in line:
            comment_lines.append(line)
        else:
            break

    return " ".join(reversed(comment_lines)) or "No comments"


def extract_code_snippet(tree, node, source_lines: List[bytes]) -> str:
    """Extract code snippet"""
    start_line = node.start_point[0]
    end_line = node.end_point[0]

    code_lines = []
    for i in range(start_line, end_line + 1):
        if i < len(source_lines):
            code_lines.append(safe_decode(source_lines[i], ""))

    return "\n".join(code_lines)


# === Text extraction utility functions ===
def get_node_text_utf8(node, source_bytes: bytes, file_path: str) -> str:
    """Safely get text from node (UTF-8 encoding)"""
    try:
        if node.start_byte is not None and node.end_byte is not None:
            node_bytes = source_bytes[node.start_byte:node.end_byte]
            return node_bytes.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        print(f"WARN: Failed to get node text: {e}")

    return "<unknown>"


# === Smart node lookup functions ===
def find_java_callee_node_id(callee_name: str, caller_class: str,
                             imports: List[str], file_path: str) -> Optional[str]:
    """Find Java callee method node ID"""

    # 1. Try fully qualified name matching
    if '.' in caller_class:
        class_method_name = f"{caller_class}.{callee_name}"
        if class_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[class_method_name]

    # 2. Try simple class name matching
    simple_class_name = caller_class.split('.')[-1] if '.' in caller_class else caller_class
    simple_method_name = f"{simple_class_name}.{callee_name}"
    if simple_method_name in FUNCTION_NODE_MAP:
        return FUNCTION_NODE_MAP[simple_method_name]

    # 3. Try method name only matching
    for key, node_id in FUNCTION_NODE_MAP.items():
        if key.endswith(f".{callee_name}"):
            return node_id

    # 4. Try import statement matching
    for import_stmt in imports:
        if import_stmt.endswith(f".{callee_name}"):
            if import_stmt in FUNCTION_NODE_MAP:
                return FUNCTION_NODE_MAP[import_stmt]

    # 5. Try containing callee method name
    for key, node_id in FUNCTION_NODE_MAP.items():
        if callee_name in key:
            return node_id

    return None


def find_cpp_callee_node_id(callee_name: str, caller_namespace: str, caller_class: str,
                            includes: List[str], file_path: str) -> Optional[str]:
    """Find C++ callee function/method node ID"""

    # 1. Try fully qualified name matching
    if caller_class:
        class_method_name = f"{caller_class}::{callee_name}"
        if class_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[class_method_name]

    # 2. Try namespace qualification
    if caller_namespace:
        namespace_method_name = f"{caller_namespace}::{callee_name}"
        if namespace_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[namespace_method_name]

    # 3. Try simple class name matching
    if caller_class:
        simple_class_name = caller_class.split('::')[-1] if '::' in caller_class else caller_class
        simple_method_name = f"{simple_class_name}::{callee_name}"
        if simple_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[simple_method_name]

    # 4. Try function name only matching
    for key, node_id in FUNCTION_NODE_MAP.items():
        if key.endswith(f"::{callee_name}"):
            return node_id

    # 5. Try inference from header includes
    for include in includes:
        # Infer possible functions from header name
        header_name = Path(include).stem
        if header_name and callee_name.lower().startswith(header_name.lower()):
            for key, node_id in FUNCTION_NODE_MAP.items():
                if callee_name in key and header_name in key:
                    return node_id

    # 6. Fuzzy matching
    for key, node_id in FUNCTION_NODE_MAP.items():
        if callee_name in key.split('::')[-1]:
            return node_id

    return None


def find_javascript_callee_node_id(callee_name: str, current_module: str = "",
                                   current_class: str = "", file_path: str = "") -> Optional[str]:
    """Find JavaScript callee function/method node ID"""

    # 1. Try fully qualified name matching
    if current_class and '.' in callee_name:
        # If callee_name is already fully qualified
        if callee_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[callee_name]

    # 2. Try class.method name matching
    if current_class:
        class_method_name = f"{current_class}.{callee_name}"
        if class_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[class_method_name]

    # 3. Try module.method name matching
    if current_module:
        module_method_name = f"{current_module}.{callee_name}"
        if module_method_name in FUNCTION_NODE_MAP:
            return FUNCTION_NODE_MAP[module_method_name]

    # 4. Try function name only matching
    for key, node_id in FUNCTION_NODE_MAP.items():
        if key.endswith(f".{callee_name}") or key == callee_name:
            return node_id

    # 5. Try to find global functions
    if callee_name in FUNCTION_NODE_MAP:
        return FUNCTION_NODE_MAP[callee_name]

    # 6. Fuzzy matching
    for key, node_id in FUNCTION_NODE_MAP.items():
        if callee_name in key.split('.')[-1]:
            return node_id

    return None


# === Fixed function definition extraction ===
def extract_function_definitions(tree, file_path: str, language: str, all_nodes: List[GraphNode], source_bytes: bytes):
    """Extract all function/method definitions, build mapping table"""
    if language == 'java':
        from tree_sitter import Node

        def traverse_node(node: Node, current_package: str = "", current_class: str = ""):
            """Recursively traverse AST nodes"""
            if not node:
                return

            node_type = node.type

            # Package declaration
            if node_type == "package_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    current_package = get_node_text_utf8(name_node, source_bytes, file_path)

            # Class declaration
            elif node_type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_name = get_node_text_utf8(name_node, source_bytes, file_path)
                    full_class_name = f"{current_package}.{class_name}" if current_package and current_package != "default" else class_name

                    # Find class body
                    for child in node.children:
                        if child.type == "class_body":
                            for member in child.children:
                                if member.type == "method_declaration":
                                    process_method_declaration(member, full_class_name)
                        else:
                            traverse_node(child, current_package, full_class_name)
                return

            # Interface declaration
            elif node_type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    interface_name = get_node_text_utf8(name_node, source_bytes, file_path)
                    full_name = f"{current_package}.{interface_name}" if current_package and current_package != "default" else interface_name

                    for child in node.children:
                        if child.type == "interface_body":
                            for member in child.children:
                                if member.type == "method_declaration":
                                    process_method_declaration(member, full_name)
                        else:
                            traverse_node(child, current_package, full_name)
                return

            # Enum declaration
            elif node_type == "enum_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    enum_name = get_node_text_utf8(name_node, source_bytes, file_path)
                    full_name = f"{current_package}.{enum_name}" if current_package and current_package != "default" else enum_name

                    for child in node.children:
                        if child.type == "enum_body":
                            for member in child.children:
                                if member.type == "method_declaration":
                                    process_method_declaration(member, full_name)
                        else:
                            traverse_node(child, current_package, full_name)
                return

            # Recursively process child nodes
            for child in node.children:
                traverse_node(child, current_package, current_class)

        def process_method_declaration(method_node: Node, class_name: str):
            """Process method declaration node"""
            if not method_node:
                return

            # Find method name
            method_name = "<unknown>"

            # Method 1: Find via declarator
            declarator = method_node.child_by_field_name("declarator")
            if declarator:
                name_node = declarator.child_by_field_name("name")
                if name_node:
                    method_name = get_node_text_utf8(name_node, source_bytes, file_path)

            # Method 2: Find identifier via traversing child nodes
            if method_name == "<unknown>":
                for child in method_node.children:
                    if child.type == "identifier":
                        method_name = get_node_text_utf8(child, source_bytes, file_path)
                        break

            if method_name == "<unknown>" or method_name in ["<clinit>", "<init>"]:
                return  # Skip constructors and static initialization blocks

            # Build full method name
            full_method_name = f"{class_name}.{method_name}"

            # Add to global mapping
            method_node_id = f"Method:{full_method_name}"
            FUNCTION_NODE_MAP[full_method_name] = method_node_id

        # Start traversal from root node
        traverse_node(tree.root_node)

    elif language == 'cpp':
        extract_cpp_function_definitions(tree, file_path, language, all_nodes, source_bytes)
    elif language == 'javascript':
        extract_javascript_function_definitions(tree, file_path, language, all_nodes, source_bytes)


def extract_cpp_function_definitions(tree, file_path: str, language: str,
                                     all_nodes: List[GraphNode], source_bytes: bytes):
    """Extract C++ function/method definitions, build mapping table"""
    if language != 'cpp':
        return

    source_code = source_bytes.decode('utf-8', errors='ignore')

    def get_node_text(node) -> str:
        try:
            if node.start_byte is not None and node.end_byte is not None:
                return source_code[node.start_byte:node.end_byte]
        except:
            pass
        return ""

    def traverse_node(node, current_namespace: str = "", current_class: str = ""):
        if not node:
            return

        node_type = node.type

        if node_type == "function_definition":
            # Extract function name
            func_name = "anonymous_function"
            for child in node.children:
                if child.type in ["function_declarator", "declarator"]:
                    for subchild in child.children:
                        if subchild.type in ["identifier", "field_identifier"]:
                            func_name = get_node_text(subchild)
                            break
                    break

            if func_name == "anonymous_function":
                return

            # Build full function name
            full_func_name = func_name
            if current_class:
                full_func_name = f"{current_class}::{func_name}"
            if current_namespace:
                full_func_name = f"{current_namespace}::{full_func_name}"

            # Add to global mapping
            node_type = NodeType.METHOD if current_class else NodeType.FUNCTION
            func_node_id = f"{node_type.value}:{full_func_name}"
            FUNCTION_NODE_MAP[full_func_name] = func_node_id

        elif node_type == "class_specifier":
            class_name = ""
            for child in node.children:
                if child.type == "type_identifier":
                    class_name = get_node_text(child)
                    break

            if class_name:
                # Process class body
                for child in node.children:
                    if child.type == "field_declaration_list":
                        for member in child.children:
                            if member.type == "function_definition":
                                traverse_node(member, current_namespace, class_name)
                    else:
                        traverse_node(child, current_namespace, class_name)

        elif node_type == "namespace_definition":
            namespace_name = ""
            for child in node.children:
                if child.type == "identifier":
                    namespace_name = get_node_text(child)
                    break

            if namespace_name:
                new_namespace = f"{current_namespace}::{namespace_name}" if current_namespace else namespace_name
                for child in node.children:
                    if child.type == "declaration_list":
                        for decl in child.children:
                            traverse_node(decl, new_namespace, "")

        # Recursively process child nodes
        for child in node.children:
            traverse_node(child, current_namespace, current_class)

    traverse_node(tree.root_node)


def extract_javascript_function_definitions(tree, file_path: str, language: str,
                                            all_nodes: List[GraphNode], source_bytes: bytes):
    """Extract all function/method definitions from JavaScript files, build mapping table"""
    if language != 'javascript':
        return

    source_code = source_bytes.decode('utf-8', errors='ignore')

    def get_node_text(node) -> str:
        try:
            if node.start_byte is not None and node.end_byte is not None:
                return source_code[node.start_byte:node.end_byte]
        except:
            pass
        return ""

    def traverse_node(node, current_class: str = "", current_module: str = ""):
        if not node:
            return

        node_type = node.type

        # Function declaration
        if node_type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = get_node_text(name_node)
                full_name = func_name
                if current_class:
                    full_name = f"{current_class}.{func_name}"
                if current_module:
                    full_name = f"{current_module}.{full_name}"

                FUNCTION_NODE_MAP[full_name] = f"Function:{full_name}"

        # Arrow function (may be assigned to variable)
        elif node_type == "arrow_function":
            # Check if parent is variable declaration
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node:
                    var_name = get_node_text(name_node)
                    full_name = var_name
                    if current_class:
                        full_name = f"{current_class}.{var_name}"
                    if current_module:
                        full_name = f"{current_module}.{full_name}"

                    FUNCTION_NODE_MAP[full_name] = f"Function:{full_name}"

        # Class method
        elif node_type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = get_node_text(name_node)
                if current_class:
                    full_name = f"{current_class}.{method_name}"
                    if current_module:
                        full_name = f"{current_module}.{full_name}"
                    FUNCTION_NODE_MAP[full_name] = f"Method:{full_name}"

        # Class declaration
        elif node_type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = get_node_text(name_node)
                for child in node.children:
                    if child.type == "class_body":
                        for member in child.children:
                            traverse_node(member, class_name, current_module)

        # Recursively process child nodes
        for child in node.children:
            traverse_node(child, current_class, current_module)

    traverse_node(tree.root_node)


# === Fixed Java file processing method (with parameter and return value node creation) ===
def process_java_file_enhanced_fixed(tree, file_path: str, source_lines: List[bytes],
                                     file_node_id: str, all_nodes: List[GraphNode],
                                     all_edges: List[GraphEdge], function_raw_info_map: Dict,
                                     enable_dataflow: bool = True):
    """Process Java file (fixed version)"""
    from tree_sitter import Node

    # Re-read source code for consistency
    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    source_code = source_bytes.decode('utf-8', errors='ignore')

    def get_node_text(node: Node) -> str:
        """Get node text"""
        try:
            if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
                if node.start_byte is not None and node.end_byte is not None:
                    return source_code[node.start_byte:node.end_byte]
        except:
            pass
        return ""

    def get_node_source_lines(node: Node) -> str:
        """Get node source code"""
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        lines = []
        for i in range(start_line, end_line + 1):
            if i < len(source_lines):
                lines.append(safe_decode(source_lines[i], file_path))
        return "\n".join(lines)

    # Extract package name
    package_name = "default"
    for child in tree.root_node.children:
        if child.type == "package_declaration":
            name_node = child.child_by_field_name("name")
            if name_node:
                package_name = get_node_text(name_node)
            break

    # Extract import statements
    imports = []
    for child in tree.root_node.children:
        if child.type == "import_declaration":
            import_text = get_node_text(child)
            imports.append(import_text)

    # Cache for collected class and method calls
    processed_methods = set()

    def process_method(method_node: Node, class_name: str, class_node_id: str,
                       file_path: str, source_lines: List[bytes], tree, imports: List[str],
                       enable_dataflow: bool):
        """Process method node (including parameter and return value node creation)"""
        # Find method name
        method_name = "<unknown>"

        # Method 1: Find via declarator
        declarator = method_node.child_by_field_name("declarator")
        if declarator:
            name_node = declarator.child_by_field_name("name")
            if name_node:
                method_name = get_node_text(name_node)

        # Method 2: Find identifier via traversing child nodes
        if method_name == "<unknown>":
            for child in method_node.children:
                if child.type == "identifier":
                    method_name = get_node_text(child)
                    break

        if method_name == "<unknown>" or method_name in ["<clinit>", "<init>"]:
            return None, None  # Skip constructors and static initialization blocks

        # Build full method name
        full_method_name = f"{class_name}.{method_name}"

        # Check if already processed
        if full_method_name in processed_methods:
            return None, None
        processed_methods.add(full_method_name)

        # Create method node
        method_node_id = f"Method:{full_method_name}"

        # Extract method signature
        signature_parts = []
        for child in method_node.children[:5]:  # Only look at first few lines
            if child.type != "block":  # Exclude method body
                signature_parts.append(get_node_text(child))

        signature = " ".join(signature_parts)[:100] + " {...}"

        # Extract return type
        return_type_node = method_node.child_by_field_name("type")
        return_type = get_node_text(return_type_node) if return_type_node else "void"

        method_graph_node = GraphNode(
            id=method_node_id,
            type=NodeType.METHOD,
            name=method_name,
            language='java',
            signature=signature,
            file_path=file_path,
            start_line=method_node.start_point[0] + 1,
            end_line=method_node.end_point[0] + 1,
            data_type=return_type
        )
        all_nodes.append(method_graph_node)

        # Add class contains edge
        all_edges.append(GraphEdge(
            source_id=class_node_id,
            target_id=method_node_id,
            type=EdgeType.CONTAINS,
            language='java',
            properties={'member_type': 'method'}
        ))

        # Store raw information
        comment = extract_comment_before(method_node, source_lines, 'java')
        code_snippet = get_node_source_lines(method_node)
        function_raw_info_map[method_node_id] = {
            'ast_node': method_node,
            'comment': comment,
            'code_snippet': code_snippet,
            'source_lines': source_lines,
            'tree': tree
        }

        # Collect method calls
        callees = collect_java_call_expressions(method_node, source_code)
        if callees:
            FUNCTION_CALL_GRAPH[full_method_name] = callees

        # Create parameter and return value nodes (if data flow analysis is enabled)
        if enable_dataflow:
            # Extract parameters
            parameters_node = method_node.child_by_field_name("parameters")
            if parameters_node:
                param_idx = 0
                for child in parameters_node.children:
                    if child.type == "formal_parameter":
                        # Extract parameter name
                        param_name_node = child.child_by_field_name("name")
                        if param_name_node:
                            param_name = get_node_text(param_name_node)
                            if param_name:
                                # Extract parameter type
                                param_type_node = child.child_by_field_name("type")
                                param_type = get_node_text(param_type_node) if param_type_node else "Object"

                                # Create parameter node
                                param_node_id = f"Param:{method_node_id}.{param_name}"
                                param_node = GraphNode(
                                    id=param_node_id,
                                    type=NodeType.PARAMETER,
                                    name=param_name,
                                    language='java',
                                    data_type=param_type,
                                    scope=method_node_id,
                                    raw_attributes={'position': param_idx}
                                )
                                all_nodes.append(param_node)

                                # Add method has parameter edge
                                all_edges.append(GraphEdge(
                                    source_id=method_node_id,
                                    target_id=param_node_id,
                                    type=EdgeType.HAS_PARAMETER,
                                    language='java',
                                    properties={'position': param_idx, 'type': param_type}
                                ))

                                param_idx += 1

            # Create return value node (if not void)
            if return_type.lower() != "void":
                return_node_id = f"Return:{method_node_id}"
                return_node = GraphNode(
                    id=return_node_id,
                    type=NodeType.RETURN_VALUE,
                    name=f"{method_name}_return",
                    language='java',
                    data_type=return_type,
                    scope=method_node_id
                )
                all_nodes.append(return_node)

                all_edges.append(GraphEdge(
                    source_id=method_node_id,
                    target_id=return_node_id,
                    type=EdgeType.RETURNS,
                    language='java',
                    properties={'type': return_type}
                ))

        return method_node_id, full_method_name

    def collect_java_call_expressions(method_node: Node, source_code: str) -> List[str]:
        """Collect call expressions in Java methods"""
        calls = set()

        def traverse_for_calls(node: Node):
            if not node:
                return

            if node.type == "method_invocation":
                # Find method name
                name_node = node.child_by_field_name("name")
                if name_node:
                    callee_name = get_node_text(name_node)
                    if callee_name and callee_name not in ["<clinit>", "<init>"]:
                        calls.add(callee_name)

            for child in node.children:
                traverse_for_calls(child)

        # Find method body
        body_node = method_node.child_by_field_name("body")
        if body_node:
            traverse_for_calls(body_node)

        return list(calls)

    def traverse_and_process(node: Node, current_class: str = "", class_node_id: str = ""):
        """Traverse and process AST nodes"""
        node_type = node.type

        # Process class declaration
        if node_type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = get_node_text(name_node)
                qualified_name = f"{package_name}.{class_name}" if package_name != "default" else class_name

                # Create class node
                new_class_node_id = f"Class:{qualified_name}"
                class_node = GraphNode(
                    id=new_class_node_id,
                    type=NodeType.CLASS,
                    name=class_name,
                    language='java',
                    signature=get_node_text(node).split('\n')[0][:100] + " {...}",
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                )
                all_nodes.append(class_node)

                # Add file contains edge
                all_edges.append(GraphEdge(
                    source_id=file_node_id,
                    target_id=new_class_node_id,
                    type=EdgeType.CONTAINS,
                    language='java'
                ))

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for member in child.children:
                            if member.type == "method_declaration":
                                process_method(member, qualified_name, new_class_node_id,
                                               file_path, source_lines, tree, imports, enable_dataflow)
                    else:
                        traverse_and_process(child, qualified_name, new_class_node_id)

                return new_class_node_id, qualified_name

        # Recursively process child nodes
        for child in node.children:
            traverse_and_process(child, current_class, class_node_id)

        return class_node_id, current_class

    # Start traversal processing
    traverse_and_process(tree.root_node)


# === C++ file processing ===
def process_cpp_file_enhanced(tree, file_path: str, source_lines: List[bytes],
                              file_node_id: str, all_nodes: List[GraphNode],
                              all_edges: List[GraphEdge], function_raw_info_map: Dict,
                              enable_dataflow: bool = True):
    """Process C++ file (enhanced version), supports cross-function data flow analysis"""
    from tree_sitter import Node

    # Re-read source code
    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    source_code = source_bytes.decode('utf-8', errors='ignore')

    def get_node_text(node: Node) -> str:
        """Get node text"""
        try:
            if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
                if node.start_byte is not None and node.end_byte is not None:
                    return source_code[node.start_byte:node.end_byte]
        except:
            pass
        return ""

    def get_node_source_lines(node: Node) -> str:
        """Get node source code"""
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        lines = []
        for i in range(start_line, end_line + 1):
            if i < len(source_lines):
                lines.append(safe_decode(source_lines[i], file_path))
        return "\n".join(lines)

    # Extract header file includes
    includes = []
    for node in tree.root_node.children:
        if node.type == "preproc_include":
            for child in node.children:
                if child.type == "string_literal" or child.type == "system_lib_string":
                    include_path = get_node_text(child).strip('"<> ')
                    includes.append(include_path)

    FILE_DEPENDENCIES[file_path] = includes

    # Collect processed functions/methods
    processed_functions = set()

    def process_function_definition(func_node: Node, class_name: str = "",
                                    namespace: str = "", is_method: bool = False):
        """Process function/method definition"""
        # Find function declarator
        declarator = None
        for child in func_node.children:
            if child.type in ["function_declarator", "declarator"]:
                declarator = child
                break

        if not declarator:
            return None, None

        # Extract function name
        func_name = "anonymous_function"
        for child in declarator.children:
            if child.type == "identifier":
                func_name = get_node_text(child)
                break
            elif child.type == "field_identifier":
                func_name = get_node_text(child)
                break
            elif child.type == "qualified_identifier":
                func_name = get_node_text(child)
                break

        if func_name == "anonymous_function":
            return None, None

        # Build full function name
        full_func_name = func_name
        if namespace:
            full_func_name = f"{namespace}::{full_func_name}"
        if class_name:
            full_func_name = f"{class_name}::{full_func_name}"

        if full_func_name in processed_functions:
            return None, None
        processed_functions.add(full_func_name)

        # Extract return type
        return_type = "void"
        for child in func_node.children:
            if child.type in ["primitive_type", "type_identifier", "qualified_identifier", "type_descriptor"]:
                return_type = get_node_text(child)
                break

        # Create function/method node
        node_type = NodeType.METHOD if is_method else NodeType.FUNCTION
        node_id = f"{node_type.value}:{full_func_name}"

        # Build function signature
        signature = get_node_text(func_node).split('{')[0].strip() + " {...}"
        if len(signature) > 200:
            signature = signature[:200] + "..."

        func_graph_node = GraphNode(
            id=node_id,
            type=node_type,
            name=func_name,
            language='cpp',
            signature=signature,
            file_path=file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            data_type=return_type
        )
        all_nodes.append(func_graph_node)

        # Add to global mapping
        FUNCTION_NODE_MAP[full_func_name] = node_id

        # If class method, add class contains edge
        if is_method and class_name:
            class_node_id = f"Class:{namespace}::{class_name}" if namespace else f"Class:{class_name}"
            all_edges.append(GraphEdge(
                source_id=class_node_id,
                target_id=node_id,
                type=EdgeType.CONTAINS,
                language='cpp',
                properties={'member_type': 'method'}
            ))
        else:
            # Global function, add file contains edge
            all_edges.append(GraphEdge(
                source_id=file_node_id,
                target_id=node_id,
                type=EdgeType.CONTAINS,
                language='cpp',
                properties={'member_type': 'function'}
            ))

        # Store raw information
        comment = extract_comment_before(func_node, source_lines, 'cpp')
        code_snippet = get_node_source_lines(func_node)
        function_raw_info_map[node_id] = {
            'ast_node': func_node,
            'comment': comment,
            'code_snippet': code_snippet,
            'source_lines': source_lines,
            'tree': tree
        }

        # Collect call relationships
        callees = collect_cpp_call_expressions(func_node, source_code)
        if callees:
            FUNCTION_CALL_GRAPH[full_func_name] = callees

        # Process parameters and returns (if data flow is enabled)
        if enable_dataflow:
            process_cpp_parameters_and_returns(func_node, full_func_name, node_id,
                                               all_nodes, all_edges, source_code)

        return node_id, full_func_name

    def process_cpp_parameters_and_returns(func_node: Node, func_name: str, func_id: str,
                                           all_nodes: List[GraphNode], all_edges: List[GraphEdge],
                                           source_code: str):
        """Process C++ function parameters and return value nodes"""
        # Find parameter list
        parameters_node = None
        for child in func_node.children:
            if child.type in ["parameter_list", "parameter_declaration_list"]:
                parameters_node = child
                break

        if parameters_node:
            param_idx = 0
            for child in parameters_node.children:
                if child.type in ["parameter_declaration", "optional_parameter_declaration"]:
                    # Extract parameter name
                    param_name_node = None
                    param_type_node = None

                    for subchild in child.children:
                        if subchild.type == "identifier":
                            param_name_node = subchild
                        elif subchild.type in ["type_identifier", "primitive_type",
                                               "qualified_identifier", "type_descriptor"]:
                            param_type_node = subchild

                    if param_name_node and param_type_node:
                        param_name = get_node_text(param_name_node)
                        param_type = get_node_text(param_type_node)

                        if param_name and param_type:
                            # Create parameter node
                            param_node_id = f"Param:{func_id}.{param_name}"
                            param_node = GraphNode(
                                id=param_node_id,
                                type=NodeType.PARAMETER,
                                name=param_name,
                                language='cpp',
                                data_type=param_type,
                                scope=func_id,
                                raw_attributes={'position': param_idx}
                            )
                            all_nodes.append(param_node)

                            # Add method has parameter edge
                            all_edges.append(GraphEdge(
                                source_id=func_id,
                                target_id=param_node_id,
                                type=EdgeType.HAS_PARAMETER,
                                language='cpp',
                                properties={'position': param_idx, 'type': param_type}
                            ))

                            param_idx += 1

        # Create return value node (if not void)
        return_type = None
        for child in func_node.children:
            if child.type in ["primitive_type", "type_identifier",
                              "qualified_identifier", "type_descriptor"]:
                return_type = get_node_text(child)
                break

        if return_type and return_type.lower() != "void":
            return_node_id = f"Return:{func_id}"
            return_node = GraphNode(
                id=return_node_id,
                type=NodeType.RETURN_VALUE,
                name=f"{func_name}_return",
                language='cpp',
                data_type=return_type,
                scope=func_id
            )
            all_nodes.append(return_node)

            all_edges.append(GraphEdge(
                source_id=func_id,
                target_id=return_node_id,
                type=EdgeType.RETURNS,
                language='cpp',
                properties={'type': return_type}
            ))

    def collect_cpp_call_expressions(func_node: Node, source_code: str) -> List[str]:
        """Collect call expressions in C++ functions"""
        calls = set()

        def traverse_for_calls(node: Node):
            if not node:
                return

            # Function call
            if node.type == "call_expression":
                # Find function name
                for child in node.children:
                    if child.type in ["identifier", "field_expression",
                                      "qualified_identifier", "qualified_call_expression"]:
                        callee_text = get_node_text(child)
                        if callee_text and "operator" not in callee_text:
                            # Clean text
                            callee_name = callee_text.split('(')[0].split('<')[0].strip()
                            if callee_name and len(callee_name) < 100:  # Reasonable function name length
                                calls.add(callee_name)
                        break
            # Method call
            elif node.type == "field_expression":
                for child in node.children:
                    if child.type == "field_identifier":
                        callee_name = get_node_text(child)
                        if callee_name and callee_name not in ["~", "operator"]:
                            calls.add(callee_name)
                        break

            for child in node.children:
                traverse_for_calls(child)

        # Find function body
        body_node = None
        for child in func_node.children:
            if child.type == "compound_statement":
                body_node = child
                break

        if body_node:
            traverse_for_calls(body_node)

        return list(calls)

    def process_class_definition(class_node: Node, namespace: str = ""):
        """Process class definition"""
        class_name = ""
        for child in class_node.children:
            if child.type == "type_identifier":
                class_name = get_node_text(child)
                break

        if not class_name:
            return

        # Build full class name
        full_class_name = class_name
        if namespace:
            full_class_name = f"{namespace}::{class_name}"

        # Create class node
        class_node_id = f"Class:{full_class_name}"
        class_graph_node = GraphNode(
            id=class_node_id,
            type=NodeType.CLASS,
            name=class_name,
            language='cpp',
            signature=get_node_text(class_node).split('\n')[0][:100] + " {...}",
            file_path=file_path,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1
        )
        all_nodes.append(class_graph_node)

        # Add file contains edge
        all_edges.append(GraphEdge(
            source_id=file_node_id,
            target_id=class_node_id,
            type=EdgeType.CONTAINS,
            language='cpp'
        ))

        # Process methods in class body
        for child in class_node.children:
            if child.type == "field_declaration_list":  # Class body
                for member in child.children:
                    if member.type == "function_definition":
                        process_function_definition(member, class_name, namespace, True)
                    elif member.type == "declaration":
                        # Process declared but not defined member functions
                        for submember in member.children:
                            if submember.type in ["function_declarator", "declarator"]:
                                # Can handle declared member functions here
                                pass

    def process_namespace(namespace_node: Node, parent_namespace: str = ""):
        """Process namespace"""
        namespace_name = ""
        for child in namespace_node.children:
            if child.type == "identifier":
                namespace_name = get_node_text(child)
                break

        if not namespace_name:
            return

        # Build full namespace
        current_namespace = f"{parent_namespace}::{namespace_name}" if parent_namespace else namespace_name

        # Process namespace body
        for child in namespace_node.children:
            if child.type == "declaration_list":
                for decl in child.children:
                    if decl.type == "function_definition":
                        process_function_definition(decl, "", current_namespace, False)
                    elif decl.type == "class_specifier":
                        process_class_definition(decl, current_namespace)
                    elif decl.type == "namespace_definition":
                        process_namespace(decl, current_namespace)

    # Main traversal logic
    def traverse_ast(node: Node, current_namespace: str = ""):
        """Traverse AST and process various declarations"""
        if not node:
            return

        node_type = node.type

        if node_type == "function_definition":
            process_function_definition(node, "", current_namespace, False)

        elif node_type == "class_specifier":
            process_class_definition(node, current_namespace)

        elif node_type == "namespace_definition":
            process_namespace(node, current_namespace)

        # Process global variable declarations (if data flow enabled)
        elif enable_dataflow and node_type in ["declaration", "init_declarator"]:
            process_global_variable(node, file_path, all_nodes, all_edges, source_code)

        # Recursively process child nodes
        for child in node.children:
            traverse_ast(child, current_namespace)

    def process_global_variable(node: Node, file_path: str, all_nodes: List[GraphNode],
                                all_edges: List[GraphEdge], source_code: str):
        """Process global variable declaration"""
        var_name = None
        var_type = None

        for child in node.children:
            if child.type in ["identifier", "field_identifier"]:
                var_name = get_node_text(child)
            elif child.type in ["type_identifier", "primitive_type",
                                "qualified_identifier", "type_descriptor"]:
                var_type = get_node_text(child)

        if var_name and var_type:
            # Create global variable node
            var_node_id = f"GlobalVar:{file_path}:{var_name}"
            var_node = GraphNode(
                id=var_node_id,
                type=NodeType.GLOBAL_VAR,
                name=var_name,
                language='cpp',
                data_type=var_type,
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                is_global=True
            )
            all_nodes.append(var_node)

            # Add to global variable mapping
            GLOBAL_VARIABLES[var_name] = {
                'node_id': var_node_id,
                'file': file_path,
                'type': var_type,
                'is_mutated': False
            }

    # Start processing AST
    traverse_ast(tree.root_node)

    return True


# === JavaScript file processing ===
def process_javascript_file_enhanced(tree, file_path: str, source_lines: List[bytes],
                                     file_node_id: str, all_nodes: List[GraphNode],
                                     all_edges: List[GraphEdge], function_raw_info_map: Dict,
                                     enable_dataflow: bool = True):
    """Process JavaScript/TypeScript file (enhanced version)"""
    from tree_sitter import Node

    with open(file_path, 'rb') as f:
        source_bytes = f.read()
    source_code = source_bytes.decode('utf-8', errors='ignore')

    def get_node_text(node: Node) -> str:
        try:
            if hasattr(node, 'start_byte') and hasattr(node, 'end_byte'):
                if node.start_byte is not None and node.end_byte is not None:
                    return source_code[node.start_byte:node.end_byte]
        except:
            pass
        return ""

    def get_node_source_lines(node: Node) -> str:
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        lines = []
        for i in range(start_line, end_line + 1):
            if i < len(source_lines):
                lines.append(safe_decode(source_lines[i], file_path))
        return "\n".join(lines)

    # JavaScript-specific AST node processing
    def process_function_definition(func_node: Node, func_type: str = "function",
                                    class_name: str = "", module_name: str = ""):
        """Process JavaScript function definition (including arrow functions, function expressions, methods, etc.)"""

        # Extract function name
        func_name = "anonymous"
        if func_type == "function_declaration":
            name_node = func_node.child_by_field_name("name")
            if name_node:
                func_name = get_node_text(name_node)
        elif func_type == "arrow_function":
            # Arrow functions may not have names
            func_name = "arrow_function"
        elif func_type == "method_definition":
            name_node = func_node.child_by_field_name("name")
            if name_node:
                func_name = get_node_text(name_node)

        # Build full identifier
        full_func_name = func_name
        if class_name:
            full_func_name = f"{class_name}.{func_name}"
        if module_name:
            full_func_name = f"{module_name}.{full_func_name}"

        # Create node
        node_type = NodeType.METHOD if class_name else NodeType.FUNCTION
        node_id = f"{node_type.value}:{full_func_name}"

        # Add to global mapping
        FUNCTION_NODE_MAP[full_func_name] = node_id

        # Create graph node
        signature = get_node_text(func_node).split('{')[0].strip() + " {...}"
        if len(signature) > 200:
            signature = signature[:200] + "..."

        func_node_obj = GraphNode(
            id=node_id,
            type=node_type,
            name=func_name,
            language='javascript',
            signature=signature,
            file_path=file_path,
            start_line=func_node.start_point[0] + 1,
            end_line=func_node.end_point[0] + 1,
            data_type="any"  # JavaScript dynamic typing
        )
        all_nodes.append(func_node_obj)

        # Add containment relationship
        if class_name:
            class_node_id = f"Class:{module_name}.{class_name}" if module_name else f"Class:{class_name}"
            all_edges.append(GraphEdge(
                source_id=class_node_id,
                target_id=node_id,
                type=EdgeType.CONTAINS,
                language='javascript',
                properties={'member_type': 'method'}
            ))
        else:
            all_edges.append(GraphEdge(
                source_id=file_node_id,
                target_id=node_id,
                type=EdgeType.CONTAINS,
                language='javascript',
                properties={'member_type': 'function'}
            ))

        # Store raw information
        comment = extract_comment_before(func_node, source_lines, 'javascript')
        code_snippet = get_node_source_lines(func_node)
        function_raw_info_map[node_id] = {
            'ast_node': func_node,
            'comment': comment,
            'code_snippet': code_snippet,
            'source_lines': source_lines,
            'tree': tree
        }

        # Collect function calls
        callees = collect_javascript_call_expressions(func_node, source_code)
        if callees:
            FUNCTION_CALL_GRAPH[full_func_name] = callees

        return node_id, full_func_name

    def collect_javascript_call_expressions(func_node: Node, source_code: str) -> List[str]:
        """Collect call expressions in JavaScript functions"""
        calls = set()

        def traverse_for_calls(node: Node):
            if not node:
                return

            # JavaScript call expressions
            if node.type == "call_expression":
                # Find called identifier
                for child in node.children:
                    if child.type in ["identifier", "member_expression"]:
                        callee_text = get_node_text(child)
                        if callee_text:
                            calls.add(callee_text.split('(')[0].strip())
                        break

            for child in node.children:
                traverse_for_calls(child)

        # Find function body
        body_node = None
        if func_node.type == "function_declaration":
            body_node = func_node.child_by_field_name("body")
        elif func_node.type == "arrow_function":
            body_node = func_node.child_by_field_name("body")
        elif func_node.type == "method_definition":
            body_node = func_node.child_by_field_name("body")

        if body_node:
            traverse_for_calls(body_node)

        return list(calls)

    def traverse_and_process(node: Node, current_class: str = "", current_module: str = ""):
        """Traverse and process AST nodes"""
        if not node:
            return

        node_type = node.type

        # Function declaration
        if node_type == "function_declaration":
            process_function_definition(node, "function_declaration",
                                        current_class, current_module)

        # Arrow function expression
        elif node_type == "arrow_function":
            process_function_definition(node, "arrow_function",
                                        current_class, current_module)

        # Class method
        elif node_type == "method_definition":
            process_function_definition(node, "method_definition",
                                        current_class, current_module)

        # Class declaration
        elif node_type == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = get_node_text(name_node)

                # Create class node
                full_class_name = f"{current_module}.{class_name}" if current_module else class_name
                class_node_id = f"Class:{full_class_name}"
                class_node = GraphNode(
                    id=class_node_id,
                    type=NodeType.CLASS,
                    name=class_name,
                    language='javascript',
                    signature=get_node_text(node).split('\n')[0][:100] + " {...}",
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                )
                all_nodes.append(class_node)

                # Add file contains edge
                all_edges.append(GraphEdge(
                    source_id=file_node_id,
                    target_id=class_node_id,
                    type=EdgeType.CONTAINS,
                    language='javascript'
                ))

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for member in child.children:
                            traverse_and_process(member, class_name, current_module)

        # ES6 module export/import
        elif node_type in ["export_statement", "import_statement"]:
            # Can handle module dependencies here
            pass

        # Recursively process child nodes
        for child in node.children:
            traverse_and_process(child, current_class, current_module)

    # Main processing logic
    traverse_and_process(tree.root_node)

    return True


# === Enhanced main file processing logic ===
def process_single_file_enhanced(file_path: str, all_nodes: List[GraphNode],
                                 all_edges: List[GraphEdge], function_raw_info_map: Dict,
                                 enable_dataflow: bool = True) -> bool:
    """Process single file (Java, C++ or JavaScript), enhanced version supports cross-function data flow"""
    language = detect_language(file_path)
    if not language:
        return False

    parser = LANGUAGE_PARSERS.get(language)
    if not parser:
        return False

    try:
        with open(file_path, 'rb') as f:
            source_bytes = f.read()
        source_lines = source_bytes.split(b'\n')
        tree = parser.parse(source_bytes)

        # Create file node
        file_node_id = f"File:{file_path}"
        file_node = GraphNode(
            id=file_node_id,
            type=NodeType.FILE,
            name=Path(file_path).name,
            language=language,
            file_path=file_path
        )
        all_nodes.append(file_node)

        # First extract all function/method definitions
        if language == 'java':
            extract_function_definitions(tree, file_path, language, all_nodes, source_bytes)
        elif language == 'cpp':
            extract_cpp_function_definitions(tree, file_path, language, all_nodes, source_bytes)
        elif language == 'javascript':
            # Added: Extract JavaScript function definitions
            extract_javascript_function_definitions(tree, file_path, language, all_nodes, source_bytes)

        # Language-specific detailed processing
        if language == 'java':
            process_java_file_enhanced_fixed(tree, file_path, source_lines, file_node_id,
                                             all_nodes, all_edges, function_raw_info_map,
                                             enable_dataflow)
        elif language == 'cpp':
            process_cpp_file_enhanced(tree, file_path, source_lines, file_node_id,
                                      all_nodes, all_edges, function_raw_info_map,
                                      enable_dataflow)
        elif language == 'javascript':
            # Added: Process JavaScript files
            process_javascript_file_enhanced(tree, file_path, source_lines, file_node_id,
                                             all_nodes, all_edges, function_raw_info_map,
                                             enable_dataflow)

        return True

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


# === Enhanced data flow analysis functions ===
def analyze_cross_function_data_flow(all_nodes: List[GraphNode], all_edges: List[GraphEdge]) -> Tuple[
    List[GraphEdge], List[Dict]]:
    """Analyze cross-function data flow (parameter passing, return value passing)"""
    new_edges = []
    flow_records = []

    # Build node lookup table
    node_dict = {node.id: node for node in all_nodes}

    # Find all call edges
    call_edges = [edge for edge in all_edges if edge.type == EdgeType.CALLS]

    for call_edge in call_edges:
        caller_id = call_edge.source_id
        callee_id = call_edge.target_id

        # Find caller and callee nodes
        caller_node = node_dict.get(caller_id)
        callee_node = node_dict.get(callee_id)

        if not caller_node or not callee_node:
            continue

        # Find caller's parameters
        caller_params = [node for node in all_nodes
                         if node.type == NodeType.PARAMETER and node.scope == caller_id]

        # Find callee's parameters
        callee_params = [node for node in all_nodes
                         if node.type == NodeType.PARAMETER and node.scope == callee_id]

        # Create parameter passing edges
        if caller_params and callee_params:
            # 1-to-1 mapping
            for i, (caller_param, callee_param) in enumerate(zip(caller_params[:len(callee_params)], callee_params)):
                new_edge = GraphEdge(
                    source_id=caller_param.id,
                    target_id=callee_param.id,
                    type=EdgeType.PASSES_TO,
                    language=caller_node.language,
                    line_number=call_edge.line_number,
                    properties={
                        'position': i,
                        'call_edge_id': f"{caller_id}->{callee_id}",
                        'caller_method': caller_node.name,
                        'callee_method': callee_node.name
                    }
                )
                new_edges.append(new_edge)
                flow_records.append({
                    'type': 'parameter_pass',
                    'caller': caller_node.name,
                    'callee': callee_node.name,
                    'position': i
                })

        # Find callee's return values
        callee_returns = [node for node in all_nodes
                          if node.type == NodeType.RETURN_VALUE and node.scope == callee_id]

        if callee_returns:
            return_node = callee_returns[0]

            # Create return value node
            return_target_id = f"Var:{caller_id}.return_value"
            return_target_node = GraphNode(
                id=return_target_id,
                type=NodeType.VARIABLE,
                name="return_value",
                language=caller_node.language,
                data_type=return_node.data_type,
                scope=caller_id
            )

            # Check if already exists
            if return_target_id not in node_dict:
                all_nodes.append(return_target_node)
                node_dict[return_target_id] = return_target_node

            # Create return value passing edge
            new_edge = GraphEdge(
                source_id=return_node.id,
                target_id=return_target_id,
                type=EdgeType.RETURNS_TO,
                language=caller_node.language,
                line_number=call_edge.line_number,
                properties={
                    'call_edge_id': f"{caller_id}->{callee_id}",
                    'caller_method': caller_node.name,
                    'callee_method': callee_node.name
                }
            )
            new_edges.append(new_edge)
            flow_records.append({
                'type': 'return_pass',
                'caller': caller_node.name,
                'callee': callee_node.name
            })

    return new_edges, flow_records


# === Project processing ===
def process_project_enhanced(project_root: str, all_nodes: List[GraphNode],
                             all_edges: List[GraphEdge], function_raw_info_map: Dict,
                             enable_dataflow: bool = True) -> Dict[str, int]:
    """Process Java/C++/JavaScript project (enhanced version)"""
    language_stats = defaultdict(int)

    for root, dirs, files in os.walk(project_root):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            language = detect_language(file_path)

            if not language:
                continue

            if process_single_file_enhanced(file_path, all_nodes, all_edges,
                                            function_raw_info_map, enable_dataflow):
                language_stats[language] += 1

    logger.info(f"File processing statistics: {dict(language_stats)}")
    return language_stats


# === C++ specific analysis enhancement ===
def enhance_cpp_analysis(all_nodes: List[GraphNode], all_edges: List[GraphEdge],
                         function_raw_info_map: Dict, source_files: Dict[str, str]):
    """Enhance C++ analysis, including call relationship building and global variable access analysis"""

    def collect_cpp_call_relations():
        """Collect C++ call relationships and build call edges"""
        call_edges_added = 0

        for caller_name, callee_names in FUNCTION_CALL_GRAPH.items():
            # Find caller node
            caller_node_id = FUNCTION_NODE_MAP.get(caller_name)
            if not caller_node_id:
                continue

            for callee_name in callee_names:
                # Try to find callee
                callee_node_id = find_cpp_callee_node_id(callee_name, "", "", [], "")

                if not callee_node_id:
                    # Try to find directly in FUNCTION_NODE_MAP
                    for key, node_id in FUNCTION_NODE_MAP.items():
                        if callee_name in key:
                            callee_node_id = node_id
                            break

                if callee_node_id:
                    # Check if same edge already exists
                    edge_exists = False
                    for edge in all_edges:
                        if (edge.source_id == caller_node_id and
                                edge.target_id == callee_node_id and
                                edge.type == EdgeType.CALLS):
                            edge_exists = True
                            break

                    if not edge_exists:
                        all_edges.append(GraphEdge(
                            source_id=caller_node_id,
                            target_id=callee_node_id,
                            type=EdgeType.CALLS,
                            language='cpp',
                            properties={
                                'caller': caller_name,
                                'callee': callee_name
                            }
                        ))
                        call_edges_added += 1

        return call_edges_added

    def analyze_cpp_global_access():
        """Analyze C++ function access to global variables"""
        global_access_edges = []

        for func_node in all_nodes:
            if func_node.language != 'cpp' or func_node.type not in [NodeType.FUNCTION, NodeType.METHOD]:
                continue

            func_name = func_node.name
            func_id = func_node.id
            file_path = func_node.file_path

            if not file_path or file_path not in source_files:
                continue

            func_text = source_files[file_path]

            # Check if each global variable is accessed in the function
            for var_name, var_info in GLOBAL_VARIABLES.items():
                if var_info.get('file') == file_path:
                    if var_name in func_text:
                        # Determine access type (read/write)
                        access_type = EdgeType.READS

                        # Check if it's a write operation
                        pattern = rf'{var_name}\s*='
                        if re.search(pattern, func_text):
                            access_type = EdgeType.WRITES
                            var_info['is_mutated'] = True

                        global_access_edges.append(GraphEdge(
                            source_id=func_id,
                            target_id=var_info['node_id'],
                            type=access_type,
                            language='cpp',
                            properties={
                                'variable': var_name,
                                'function': func_name
                            }
                        ))

        return global_access_edges

    # 1. Build call relationships
    logger.info("Building C++ call graph...")
    cpp_call_edges = collect_cpp_call_relations()

    # 2. Analyze global variable access
    logger.info("Analyzing C++ global variable access...")
    global_access_edges = analyze_cpp_global_access()

    all_edges.extend(global_access_edges)

    # 3. Record global variable statistics
    global_vars = {k: v for k, v in GLOBAL_VARIABLES.items()}

    return {
        'call_edges_added': cpp_call_edges,
        'global_access_edges': len(global_access_edges),
        'global_variables_count': len(global_vars)
    }


# === JavaScript specific analysis enhancement ===
def enhance_javascript_analysis(all_nodes: List[GraphNode], all_edges: List[GraphEdge],
                                function_raw_info_map: Dict, source_files: Dict[str, str]):
    """Enhance JavaScript analysis, including call relationship building"""

    def collect_javascript_call_relations():
        """Collect JavaScript call relationships and build call edges"""
        call_edges_added = 0

        for caller_name, callee_names in FUNCTION_CALL_GRAPH.items():
            # Find caller node
            caller_node_id = FUNCTION_NODE_MAP.get(caller_name)
            if not caller_node_id:
                continue

            for callee_name in callee_names:
                # Try to find callee
                callee_node_id = find_javascript_callee_node_id(callee_name, "", "", "")

                if not callee_node_id:
                    # Try to find directly in FUNCTION_NODE_MAP
                    for key, node_id in FUNCTION_NODE_MAP.items():
                        if callee_name in key:
                            callee_node_id = node_id
                            break

                if callee_node_id:
                    # Check if same edge already exists
                    edge_exists = False
                    for edge in all_edges:
                        if (edge.source_id == caller_node_id and
                                edge.target_id == callee_node_id and
                                edge.type == EdgeType.CALLS):
                            edge_exists = True
                            break

                    if not edge_exists:
                        all_edges.append(GraphEdge(
                            source_id=caller_node_id,
                            target_id=callee_node_id,
                            type=EdgeType.CALLS,
                            language='javascript',
                            properties={
                                'caller': caller_name,
                                'callee': callee_name
                            }
                        ))
                        call_edges_added += 1

        return call_edges_added

    def analyze_javascript_module_dependencies():
        """Analyze JavaScript module dependencies"""
        module_edges = []

        for file_path, source_code in source_files.items():
            if detect_language(file_path) != 'javascript':
                continue

            # Analyze ES6 module imports
            import_patterns = [
                r"import\s+(?:\{.*?\}|\*.*?|\w+)\s+from\s+['\"]([^'\"]+)['\"]",
                r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
            ]

            for pattern in import_patterns:
                matches = re.findall(pattern, source_code, re.DOTALL)
                for match in matches:
                    if isinstance(match, tuple):
                        imported_module = match[0] if match else str(match)
                    else:
                        imported_module = match
                    
                    # Create dependency edge
                    module_edges.append(GraphEdge(
                        source_id=f"File:{file_path}",
                        target_id=f"Module:{imported_module}",
                        type=EdgeType.IMPORT,
                        language='javascript',
                        properties={'import_path': imported_module}
                    ))

        return module_edges

    # 1. Build call relationships
    logger.info("Building JavaScript call graph...")
    js_call_edges = collect_javascript_call_relations()

    # 2. Analyze module dependencies
    logger.info("Analyzing JavaScript module dependencies...")
    module_edges = analyze_javascript_module_dependencies()

    all_edges.extend(module_edges)

    return {
        'call_edges_added': js_call_edges,
        'module_edges_added': len(module_edges)
    }


# === Main processing ===
def process_directory_enhanced(source_dir: str, output_dir: str = "./output",
                               enable_dataflow: bool = True) -> Dict:
    """Main processing: analyze Java/C++/JavaScript project, enhanced version supports cross-function data flow"""
    all_nodes = []
    all_edges = []
    function_raw_info_map = {}
    
    # 1. Process all source files
    logger.info("Starting project processing...")
    language_stats = process_project_enhanced(source_dir, all_nodes, all_edges, 
                                              function_raw_info_map, enable_dataflow)
    
    # 2. Build call relationships
    logger.info("Building call graphs...")
    
    # 2.1 Collect Java call relationships
    java_call_edges = 0
    for caller_name, callee_names in FUNCTION_CALL_GRAPH.items():
        caller_node_id = FUNCTION_NODE_MAP.get(caller_name)
        if not caller_node_id:
            continue
            
        for callee_name in callee_names:
            # Try to find callee
            callee_node_id = None
            
            if "java" in caller_node_id.lower():
                # Java: try to find in current class or imported classes
                pass
            elif "cpp" in caller_node_id.lower():
                # C++: handle namespace and class
                pass
            elif "javascript" in caller_node_id.lower():
                # JavaScript: handle module and class
                pass
                
            if callee_node_id:
                # Check if edge already exists
                edge_exists = any(e for e in all_edges 
                                  if e.source_id == caller_node_id and 
                                  e.target_id == callee_node_id and 
                                  e.type == EdgeType.CALLS)
                
                if not edge_exists:
                    all_edges.append(GraphEdge(
                        source_id=caller_node_id,
                        target_id=callee_node_id,
                        type=EdgeType.CALLS,
                        language='java',
                        properties={'caller': caller_name, 'callee': callee_name}
                    ))
                    java_call_edges += 1
    
    # 3. Data flow analysis
    if enable_dataflow:
        logger.info("Performing cross-function data flow analysis...")
        data_flow_edges, flow_records = analyze_cross_function_data_flow(all_nodes, all_edges)
        all_edges.extend(data_flow_edges)
        
        logger.info(f"Data flow analysis: {len(data_flow_edges)} edges, {len(flow_records)} records")
    
    # 4. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Save results
    output_files = {}
    
    # 5.1 Save nodes
    nodes_file = os.path.join(output_dir, "nodes.json")
    with open(nodes_file, "w", encoding="utf-8") as f:
        json.dump([node.dict() for node in all_nodes], f, ensure_ascii=False, indent=2)
    output_files["nodes"] = nodes_file
    
    # 5.2 Save edges
    edges_file = os.path.join(output_dir, "edges.json")
    with open(edges_file, "w", encoding="utf-8") as f:
        json.dump([edge.dict() for edge in all_edges], f, ensure_ascii=False, indent=2)
    output_files["edges"] = edges_file
    
    # 5.3 Save statistics
    stats = {
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "language_statistics": dict(language_stats),
        "function_count": len(function_raw_info_map),
        "call_relations_count": len(FUNCTION_CALL_GRAPH),
        "global_variables": len(GLOBAL_VARIABLES),
        "enable_dataflow": enable_dataflow
    }
    
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    output_files["statistics"] = stats_file
    
    # 6. Log summary
    logger.info(f"Processing complete: {len(all_nodes)} nodes, {len(all_edges)} edges")
    logger.info(f"Function nodes: {len([n for n in all_nodes if n.type in [NodeType.FUNCTION, NodeType.METHOD]])}")
    logger.info(f"Class nodes: {len([n for n in all_nodes if n.type == NodeType.CLASS])}")
    logger.info(f"Call edges: {len([e for e in all_edges if e.type == EdgeType.CALLS])}")
    
    if enable_dataflow:
        logger.info(f"Data flow edges: {len([e for e in all_edges if e.type in [EdgeType.PASSES_TO, EdgeType.RETURNS_TO]])}")
    
    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "function_info": function_raw_info_map,
        "statistics": stats,
        "output_files": output_files
    }


# === Enhanced LLM interaction functions ===
def summarize_function_with_llm_enhanced(function_node: GraphNode, raw_info: Dict) -> str:
    """Generate function summary using LLM (enhanced version)"""
    try:
        code_snippet = raw_info.get('code_snippet', '')
        comment = raw_info.get('comment', '')
        
        if not code_snippet or code_snippet.strip() == '':
            return "No code available for summarization"
        
        prompt = f"""
Please analyze the following {function_node.language} function and provide a concise summary:

Function Name: {function_node.name}
Function Signature: {function_node.signature or 'N/A'}
Location: {function_node.file_path}:{function_node.start_line}-{function_node.end_line}
Previous Comments: {comment[:500] if comment else 'No comments'}

Function Code:{function_node.language}
{code_snippet[:2000]}
Please provide:
1. Brief description of what the function does
2. Main logic flow
3. Key parameters and their purposes
4. Return value meaning
5. Any notable side effects or error handling

Output format (English only):
Summary: [1-2 sentence summary]
Logic: [brief logic flow]
Parameters: [list parameters with descriptions]
Returns: [describe return value]
Notes: [any additional notes]
"""
        
        return f"Function: {function_node.name} - Generated summary available"
        
    except Exception as e:
        logger.error(f"LLM summarization error: {e}")
        return f"Summary generation failed: {str(e)}"


def enhance_all_functions_with_llm(all_nodes: List[GraphNode], function_raw_info_map: Dict) -> List[GraphNode]:
    """Generate summaries for all functions using LLM"""
    enhanced_nodes = []
    
    for node in all_nodes:
        if node.type in [NodeType.FUNCTION, NodeType.METHOD]:
            raw_info = function_raw_info_map.get(node.id)
            if raw_info:
                summary = summarize_function_with_llm_enhanced(node, raw_info)
                node.summary = summary
                enhanced_nodes.append(node)
    
    return enhanced_nodes


# === Main entry point ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logger.error("Please specify source directory")
        logger.info(f"Usage: python {sys.argv[0]} <source_directory> [output_directory] [--enable-dataflow]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    enable_dataflow = "--enable-dataflow" in sys.argv
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    try:
        logger.info("=" * 60)
        logger.info(f"Java/C++/JavaScript Enhanced Analysis Tool")
        logger.info(f"Source directory: {source_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Data flow analysis: {enable_dataflow}")
        logger.info("=" * 60)
        
        start_time = time.time()
        result = process_directory_enhanced(source_dir, output_dir, enable_dataflow)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
        # Generate summaries
        logger.info("Generating function summaries...")
        enhanced_nodes = enhance_all_functions_with_llm(result['nodes'], result['function_info'])
        logger.info(f"Generated summaries for {len(enhanced_nodes)} functions")
        
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)