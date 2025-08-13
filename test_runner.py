"""
Test runner for RAGFlow gRPC tests.
"""
import sys
import os
import asyncio
import pytest
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.conftest import is_grpc_server_available


def run_unit_tests():
    """Run unit tests only."""
    print("Running unit tests...")
    return pytest.main([
        "tests/unit/",
        "-v",
        "-m", "unit",
        "--tb=short"
    ])


def run_grpc_tests():
    """Run gRPC tests."""
    print("Checking gRPC server availability...")
    
    if not is_grpc_server_available():
        print("gRPC server not available. Please start the gRPC server first.")
        return 1
    
    print("gRPC server is available")
    print("Running gRPC tests...")
    return pytest.main([
        "tests/",
        "-v",
        "-m", "grpc",
        "--tb=short"
    ])


def run_all_tests():
    """Run all tests."""
    print("Running all tests...")
    
    # Run unit tests first
    unit_result = run_unit_tests()
    
    # Run gRPC tests if unit tests pass
    if unit_result == 0:
        grpc_result = run_grpc_tests()
        return grpc_result
    else:
        return unit_result


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")
    
    # Check if protobuf files exist
    proto_files = [
        "grpc_ragflow_server/ragflow_service_pb2.py",
        "grpc_ragflow_server/ragflow_service_pb2_grpc.py"
    ]
    
    for proto_file in proto_files:
        if not os.path.exists(os.path.join(Path(__file__).parent, proto_file)):
            print(f" Missing protobuf file: {proto_file}")
            print("Run 'make protobuf' or generate protobuf files manually")
            return False
    
    print(" Protobuf files are present")
    
    # Check if example data exists
    example_data_dir = "example_data/documents"
    if not os.path.exists(os.path.join(Path(__file__).parent, example_data_dir)):
        print(f"Missing example data directory: {example_data_dir}")
        return False
    
    print("Example data directory exists")
    
    # Check service availability
    grpc_available = is_grpc_server_available()
    
    print(f"gRPC server available: {'yes' if grpc_available else 'No'}")
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAGFlow gRPC Test Runner")
    parser.add_argument(
        "test_type",
        choices=["unit", "grpc", "all", "check"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--markers",
        help="Pytest markers to filter tests (e.g., 'not slow')"
    )
    parser.add_argument(
        "--pattern",
        help="Test file pattern to match"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    if args.test_type == "check":
        success = check_test_environment()
        return 0 if success else 1
    
    # Check environment before running tests
    if not check_test_environment():
        print(" Environment check failed. Please fix the issues above.")
        return 1
    
    # Build pytest arguments
    pytest_args = []
    
    if args.test_type == "unit":
        pytest_args.extend(["tests/unit/", "-m", "unit"])
    elif args.test_type == "grpc":
        pytest_args.extend(["tests/", "-m", "grpc"])
    elif args.test_type == "all":
        pytest_args.append("tests/")
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    if args.pattern:
        pytest_args.extend(["-k", args.pattern])
    
    pytest_args.append("--tb=short")
    
    print(f"Running pytest with args: {' '.join(pytest_args)}")
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
