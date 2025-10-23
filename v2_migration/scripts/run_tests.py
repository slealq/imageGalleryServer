"""
Test runner script with options.

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py --unit       # Run unit tests only
    python scripts/run_tests.py --integration  # Run integration tests only
    python scripts/run_tests.py --cov        # Run with coverage report
"""
import subprocess
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(args: list[str] = None):
    """Run pytest with given arguments."""
    if args is None:
        args = []
    
    cmd = ["pytest"] + args
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    # Quick shortcuts
    if "--unit" in args:
        args.remove("--unit")
        args.extend(["tests/unit", "-m", "unit or not integration"])
    
    if "--integration" in args:
        args.remove("--integration")
        args.extend(["tests/integration"])
    
    if "--cov" in args:
        args.remove("--cov")
        args.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if "--fast" in args:
        args.remove("--fast")
        args.append("-x")  # Stop on first failure
    
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())






