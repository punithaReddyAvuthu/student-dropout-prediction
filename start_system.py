import subprocess
import time
import sys
import os
import requests
import signal

def start_backend():
    print("🚀 Starting Backend API (Flask)...")
    api_path = os.path.join(os.path.dirname(__file__), 'api', 'app.py')
    
    # Open log file to capture output without blocking (deadlocking) the process
    log_file = open("api_log.txt", "a", encoding="utf-8")
    log_file.write(f"\n--- System Start: {time.ctime()} ---\n")
    
    # Run in a new session and redirect output to file
    process = subprocess.Popen([sys.executable, api_path], 
                             stdout=log_file, 
                             stderr=subprocess.STDOUT, 
                             text=True,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    return process

def wait_for_api(timeout=600):
    print("⏳ Waiting for API to be ready...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:5000/", timeout=10)
            if response.status_code == 200:
                print("✅ Backend API is ONLINE!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def start_dashboard():
    print("🎨 Starting Streamlit Dashboard...")
    dash_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    try:
        subprocess.run(["streamlit", "run", dash_path])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

def main():
    api_process = None
    try:
        # 1. Start API
        api_process = start_backend()
        
        # 2. Wait for API
        if not wait_for_api():
            print("❌ Error: API failed to start within timeout. Check api/app.py for errors.")
            if api_process:
                api_process.terminate()
            return

        # 3. Start Dashboard
        start_dashboard()
        
    finally:
        # 4. Cleanup
        if api_process:
            print("🧹 Cleaning up background processes...")
            if os.name == 'nt':
                # On Windows, need taskkill to ensure sub-processes like watchdog are killed
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(api_process.pid)], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                api_process.terminate()
            print("✨ All systems shut down.")

if __name__ == "__main__":
    main()
