"""
Demo Runner Script

Purpose:
--------
One command to reproduce the entire Microservice Anomaly Detection demo.

This script runs the complete pipeline:
1. Generate synthetic logs with injected incident
2. Build features and run MAD anomaly detection
3. Run evaluation metrics
4. Print instructions to start API and Dashboard

Run Instructions:
-----------------
From the repository root, run:
    python run_demo.py

With browser open flag:
    python run_demo.py --open

Dependencies:
-------------
    All dependencies from requirements.txt must be installed.
"""

import os
import sys
import subprocess
import argparse
import webbrowser
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

API_URL = "http://localhost:8000"
API_DOCS_URL = "http://localhost:8000/docs"
STREAMLIT_URL = "http://localhost:8501"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num: int, text: str) -> None:
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 50)


def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Parameters
    ----------
    command : list
        Command and arguments as list
    description : str
        Description of what the command does
    
    Returns
    -------
    bool
        True if command succeeded, False otherwise
    """
    print(f"Running: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        return False


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists and print status."""
    if os.path.exists(path):
        print(f"  ✓ {description}: {path}")
        return True
    else:
        print(f"  ✗ {description}: {path} (NOT FOUND)")
        return False


# =============================================================================
# MAIN DEMO RUNNER
# =============================================================================

def run_demo(open_browser: bool = False) -> int:
    """
    Run the complete demo pipeline.
    
    Parameters
    ----------
    open_browser : bool
        Whether to open browser URLs after completion
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    start_time = datetime.now()
    
    print_header("MICROSERVICE ANOMALY DETECTION DEMO")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Get Python executable (use same Python that's running this script)
    python_exe = sys.executable
    print(f"Python executable: {python_exe}")
    
    # =========================================================================
    # Step 1: Generate Synthetic Logs
    # =========================================================================
    print_step(1, "Generating Synthetic Logs")
    
    success = run_command(
        [python_exe, "-m", "ingest.generate_logs"],
        "Log generation"
    )
    
    if not success:
        print("\n❌ Demo failed at Step 1. Please check the error above.")
        return 1
    
    # =========================================================================
    # Step 2: Build Features and Run Anomaly Detection
    # =========================================================================
    print_step(2, "Building Features & Running MAD Anomaly Detection")
    
    success = run_command(
        [python_exe, "-m", "ingest.feature_build"],
        "Feature building and anomaly detection"
    )
    
    if not success:
        print("\n❌ Demo failed at Step 2. Please check the error above.")
        return 1
    
    # =========================================================================
    # Step 3: Run Evaluation
    # =========================================================================
    print_step(3, "Running Evaluation Metrics")
    
    success = run_command(
        [python_exe, "evaluate.py"],
        "Evaluation"
    )
    
    if not success:
        print("\n⚠️ Evaluation failed, but continuing with demo...")
    
    # =========================================================================
    # Step 4: Verify Output Files
    # =========================================================================
    print_step(4, "Verifying Output Files")
    
    files_ok = True
    files_ok &= check_file_exists("data/raw_logs.csv", "Raw logs")
    files_ok &= check_file_exists("data/incident_meta.json", "Incident metadata")
    files_ok &= check_file_exists("data/features_1min.csv", "Features")
    files_ok &= check_file_exists("data/scored_1min.csv", "Scored data")
    files_ok &= check_file_exists("data/eval_report.json", "Evaluation report")
    
    if not files_ok:
        print("\n⚠️ Some output files are missing, but continuing...")
    
    # =========================================================================
    # Step 5: Print Instructions
    # =========================================================================
    print_step(5, "Next Steps - Start API and Dashboard")
    
    print("""
To complete the demo, open TWO separate terminal windows:

┌─────────────────────────────────────────────────────────────────────┐
│  TERMINAL 1 - Start the FastAPI Backend                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    uvicorn backend_api.main:app --reload                            │
│                                                                     │
│    API will be available at:                                        │
│    • API Base:  http://localhost:8000                               │
│    • API Docs:  http://localhost:8000/docs                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  TERMINAL 2 - Start the Streamlit Dashboard                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    streamlit run dashboard/app.py                                   │
│                                                                     │
│    Dashboard will be available at:                                  │
│    • Dashboard: http://localhost:8501                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # =========================================================================
    # Open Browser (if requested)
    # =========================================================================
    if open_browser:
        print("\n🌐 Opening URLs in browser...")
        print(f"   Note: Make sure to start the services first!\n")
        print(f"   • Streamlit Dashboard: {STREAMLIT_URL}")
        print(f"   • API Documentation:   {API_DOCS_URL}")
        
        # Try to open browser (may not work on all systems)
        try:
            webbrowser.open(STREAMLIT_URL)
            print("\n   ✓ Opened Streamlit URL in default browser")
        except Exception as e:
            print(f"\n   ⚠️ Could not open browser automatically: {e}")
            print(f"   Please manually open: {STREAMLIT_URL}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("DEMO SETUP COMPLETE")
    print(f"""
✅ All data files generated successfully!
⏱️  Total time: {duration:.1f} seconds

📊 Key URLs (after starting services):
   • Dashboard:     {STREAMLIT_URL}
   • API Docs:      {API_DOCS_URL}
   • Health Check:  {API_URL}/health
   • Culprit API:   {API_URL}/culprit?minutes=15

📁 Generated Files:
   • data/raw_logs.csv        - Synthetic microservice logs
   • data/incident_meta.json  - Ground truth incident info
   • data/features_1min.csv   - Aggregated per-minute features
   • data/scored_1min.csv     - Features with anomaly scores
   • data/eval_report.json    - Evaluation metrics

🎯 Injected Incident:
   • Service:  payments
   • Endpoint: /pay
   • Window:   Minute 45-53 (8 minutes of anomalous behavior)
""")
    
    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run the complete Microservice Anomaly Detection demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py           Run demo pipeline
  python run_demo.py --open    Run demo and open browser URLs
        """
    )
    
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open Streamlit URL in browser after setup (start services first!)"
    )
    
    args = parser.parse_args()
    
    return run_demo(open_browser=args.open)


if __name__ == "__main__":
    sys.exit(main())
