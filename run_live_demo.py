"""
Live Demo Runner

One-stop script to run the complete live anomaly detection demo.

Usage:
------
    # Just print instructions:
    python run_live_demo.py

    # Launch specific components:
    python run_live_demo.py --api          # Start FastAPI server
    python run_live_demo.py --sim          # Start event simulator
    python run_live_demo.py --ui           # Start Streamlit dashboard
    python run_live_demo.py --all          # Start all (requires 3 terminals)

Note:
-----
    For best results, run each component in a separate terminal.
    This script can launch subprocesses, but separate terminals are recommended.
"""

import argparse
import subprocess
import sys
import os
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

API_CMD = "uvicorn backend_api.main:app --reload --host 127.0.0.1 --port 8000"
SIM_CMD = "python -m simulate.live_sender"
UI_CMD = "streamlit run dashboard/app.py"

API_URL = "http://127.0.0.1:8000"
UI_URL = "http://localhost:8501"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("  🎯 Microservice Anomaly Detection - LIVE DEMO")
    print("=" * 70)
    print()


def print_instructions():
    """Print step-by-step instructions."""
    print("📋 STEP-BY-STEP INSTRUCTIONS")
    print("-" * 70)
    print()
    print("Open 3 separate terminals and run these commands:")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ TERMINAL 1: Start the API Server                                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  {API_CMD}")
    print("│                                                                     │")
    print("│  Wait for: 'Uvicorn running on http://127.0.0.1:8000'              │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ TERMINAL 2: Start the Event Simulator                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  {SIM_CMD}")
    print("│                                                                     │")
    print("│  This sends ~25 events/sec. Incident starts at 30s, lasts 60s.     │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ TERMINAL 3: Start the Dashboard                                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  {UI_CMD}")
    print("│                                                                     │")
    print("│  Open browser to http://localhost:8501                             │")
    print("│  Enable 'Live Mode' toggle in sidebar!                             │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    print("-" * 70)
    print("🎬 DEMO FLOW:")
    print("   0-30s:   Normal traffic (green metrics)")
    print("   30-90s:  INCIDENT on payments:/pay (latency spike + errors)")
    print("   90s+:    Recovery (metrics normalize)")
    print()
    print("👀 WHAT TO WATCH:")
    print("   - Dashboard shows 'Top Culprit' as payments:/pay")
    print("   - Severity changes from Info → Warning → Critical")
    print("   - Click 'Explain This Incident' for AI-style explanation")
    print("-" * 70)
    print()


def launch_subprocess(cmd: str, name: str, wait: bool = False) -> subprocess.Popen:
    """
    Launch a subprocess.
    
    Parameters
    ----------
    cmd : str
        Command to run
    name : str
        Display name for logging
    wait : bool
        Whether to wait for process to complete
    
    Returns
    -------
    subprocess.Popen
        Process handle (or None if wait=True)
    """
    print(f"🚀 Starting {name}...")
    print(f"   Command: {cmd}")
    
    try:
        if sys.platform == "win32":
            # Windows: use start to open new terminal
            # This is more reliable than running in background
            process = subprocess.Popen(
                cmd,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if not wait else 0
            )
        else:
            # Unix: run in background
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE if not wait else None,
                stderr=subprocess.PIPE if not wait else None
            )
        
        if wait:
            process.wait()
        else:
            time.sleep(1)  # Give it a moment to start
            if process.poll() is not None:
                print(f"   ⚠️  Process exited immediately (code: {process.returncode})")
            else:
                print(f"   ✓ {name} started (PID: {process.pid})")
        
        return process
    
    except Exception as e:
        print(f"   ❌ Failed to start {name}: {e}")
        return None


def check_api_ready(timeout: int = 10) -> bool:
    """Check if API is responding."""
    import urllib.request
    
    print(f"⏳ Waiting for API to be ready (max {timeout}s)...")
    
    for i in range(timeout):
        try:
            response = urllib.request.urlopen(f"{API_URL}/health", timeout=1)
            if response.status == 200:
                print("   ✓ API is ready!")
                return True
        except Exception:
            pass
        time.sleep(1)
        print(f"   Waiting... ({i+1}/{timeout})")
    
    print("   ⚠️  API not responding (continuing anyway)")
    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the Microservice Anomaly Detection live demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live_demo.py              # Print instructions only
  python run_live_demo.py --api        # Start API server
  python run_live_demo.py --sim        # Start simulator
  python run_live_demo.py --ui         # Start dashboard
  python run_live_demo.py --all        # Start all components
        """
    )
    
    parser.add_argument("--api", action="store_true", help="Launch API server")
    parser.add_argument("--sim", action="store_true", help="Launch event simulator")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Launch all components")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print_banner()
    
    # If no launch flags, just print instructions
    if not (args.api or args.sim or args.ui or args.all):
        print_instructions()
        print("💡 TIP: Use --api, --sim, --ui, or --all to launch components")
        print("        (Or copy commands above into separate terminals)")
        print()
        return
    
    # Launch requested components
    processes = []
    
    if args.api or args.all:
        proc = launch_subprocess(API_CMD, "API Server")
        if proc:
            processes.append(("API", proc))
        
        if args.all:
            # Wait for API to be ready before starting others
            check_api_ready(timeout=15)
    
    if args.sim or args.all:
        proc = launch_subprocess(SIM_CMD, "Event Simulator")
        if proc:
            processes.append(("Simulator", proc))
    
    if args.ui or args.all:
        proc = launch_subprocess(UI_CMD, "Streamlit Dashboard")
        if proc:
            processes.append(("Dashboard", proc))
    
    print()
    print("-" * 70)
    
    if processes:
        print(f"✓ Started {len(processes)} component(s)")
        print()
        print("📍 URLs:")
        print(f"   API Docs:  {API_URL}/docs")
        print(f"   Dashboard: {UI_URL}")
        print()
        print("⚠️  Press Ctrl+C to stop all processes")
        print("-" * 70)
        
        # Wait for interrupt
        try:
            while True:
                time.sleep(1)
                # Check if any process died
                for name, proc in processes:
                    if proc.poll() is not None:
                        print(f"⚠️  {name} exited (code: {proc.returncode})")
        except KeyboardInterrupt:
            print("\n🛑 Stopping all processes...")
            for name, proc in processes:
                try:
                    proc.terminate()
                    print(f"   Stopped {name}")
                except Exception:
                    pass
    else:
        print("⚠️  No processes were started successfully.")
        print("    Try running the commands manually in separate terminals.")
        print()
        print_instructions()


if __name__ == "__main__":
    main()
