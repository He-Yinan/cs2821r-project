#!/usr/bin/env python3
"""
Test MARA-RAG installation and verify all components are importable.

Run this script to verify that MARA-RAG is correctly installed and
all dependencies are available.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))


def test_imports():
    """Test that all required modules can be imported."""

    print("Testing MARA-RAG imports...")
    print()

    tests = []

    # Test 1: Import HippoRAG
    try:
        from hipporag.HippoRAG import HippoRAG
        from hipporag.utils.config_utils import BaseConfig
        print("✓ HippoRAG imports successful")
        tests.append(True)
    except ImportError as e:
        print(f"✗ HippoRAG import failed: {e}")
        tests.append(False)

    # Test 2: Import numpy and scipy
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        print("✓ NumPy and SciPy imports successful")
        tests.append(True)
    except ImportError as e:
        print(f"✗ NumPy/SciPy import failed: {e}")
        tests.append(False)

    # Test 3: Import igraph
    try:
        import igraph as ig
        print("✓ igraph import successful")
        tests.append(True)
    except ImportError as e:
        print(f"✗ igraph import failed: {e}")
        tests.append(False)

    # Test 4: Import MARA-RAG components
    try:
        from query_router import QueryRouter
        from relation_aware_ppr import RelationAwarePPR
        print("✓ MARA-RAG components import successful")
        tests.append(True)
    except ImportError as e:
        print(f"✗ MARA-RAG components import failed: {e}")
        tests.append(False)

    # Test 5: Import evaluation modules
    try:
        from hipporag.evaluation.retrieval_eval import RetrievalRecall
        from hipporag.evaluation.qa_eval import QAF1Score, QAExactMatch
        print("✓ Evaluation modules import successful")
        tests.append(True)
    except ImportError as e:
        print(f"✗ Evaluation modules import failed: {e}")
        tests.append(False)

    print()
    print("="*60)

    if all(tests):
        print("✓ All tests passed! MARA-RAG is ready to use.")
        return 0
    else:
        print(f"✗ {sum(not t for t in tests)} test(s) failed.")
        print()
        print("Please ensure all dependencies are installed:")
        print("  - HippoRAG and its dependencies")
        print("  - scipy (for sparse matrices)")
        print("  - igraph (for graph operations)")
        return 1


def test_relation_classifier():
    """Test the RelationTypeClassifier."""

    print()
    print("Testing RelationTypeClassifier...")
    print()

    try:
        from graph_preprocessing import RelationTypeClassifier

        classifier = RelationTypeClassifier()

        # Test cases
        test_cases = [
            ("HIERARCHICAL", "hierarchical"),
            ("TEMPORAL", "temporal"),
            ("SPATIAL", "spatial"),
            ("CAUSALITY", "causality"),
            ("ATTRIBUTION", "attribution"),
            ("SYNONYMY", "synonym"),
            ("PRIMARY", "passage"),
            ("SECONDARY", "passage"),
            ("UNKNOWN", "other"),
        ]

        all_passed = True
        for input_type, expected_output in test_cases:
            result = classifier.classify_edge(input_type)
            status = "✓" if result == expected_output else "✗"
            if result != expected_output:
                all_passed = False
            print(f"  {status} classify_edge('{input_type}') = '{result}' "
                  f"(expected: '{expected_output}')")

        print()
        if all_passed:
            print("✓ RelationTypeClassifier tests passed")
            return True
        else:
            print("✗ Some RelationTypeClassifier tests failed")
            return False

    except Exception as e:
        print(f"✗ RelationTypeClassifier test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are present."""

    print()
    print("Testing file structure...")
    print()

    required_files = [
        "graph_preprocessing.py",
        "query_router.py",
        "relation_aware_ppr.py",
        "run_mara_experiment.py",
        "utils.py",
        "compare_results.py",
        "README.md",
        "__init__.py",
    ]

    online_retrieval_dir = Path(__file__).parent

    all_present = True
    for filename in required_files:
        filepath = online_retrieval_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")
            all_present = False

    print()
    if all_present:
        print("✓ All required files present")
        return True
    else:
        print("✗ Some required files are missing")
        return False


def main():
    """Run all tests."""

    print("="*60)
    print("MARA-RAG Installation Test")
    print("="*60)
    print()

    results = []

    # Test imports
    results.append(test_imports() == 0)

    # Test file structure
    results.append(test_file_structure())

    # Test relation classifier
    results.append(test_relation_classifier())

    # Summary
    print()
    print("="*60)
    print("Test Summary")
    print("="*60)
    print()

    if all(results):
        print("✓ All installation tests passed!")
        print()
        print("Next steps:")
        print("  1. Run offline indexing (scripts 01-05) to build a HippoRAG graph")
        print("  2. Run graph_preprocessing.py to build relation matrices")
        print("  3. Run run_mara_experiment.py to evaluate MARA-RAG")
        print()
        print("See README.md for detailed instructions.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
