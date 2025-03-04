import traceback

def handle_exception(e):
    """Handles exceptions by logging traceback and returning error details."""
    traceback_str = traceback.format_exc()
    print(traceback_str) 
    line_no = traceback.extract_tb(e.__traceback__)[-1][1]
    print(f"Exception occurred on line {line_no}")
    return str(e)
