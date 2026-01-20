"""
FileNo Extractor - Configuration
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CASES_DIR = os.path.join(BASE_DIR, "test_cases")
INCLUDE_DIR = os.path.join(TEST_CASES_DIR, "include")

# Target function to find
TARGET_FUNCTION = "mpf_mfs_open"
TARGET_ARG_INDEX = 2  # 3rd argument (0-indexed)

# Clang args for parsing
CLANG_ARGS = [
    f"-I{INCLUDE_DIR}",
    "-ferror-limit=0",
    "-x", "c",
]
