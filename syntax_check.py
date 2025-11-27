
try:
    from main import app
    print("✅ main.py imported successfully. Syntax is correct.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
