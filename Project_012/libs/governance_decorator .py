import requests
import functools

def governance_guard(endpoint="http://governance:8001/scan"):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Extract input from kwargs (or args convention)
            user_input = kwargs.get("query") or kwargs.get("user_input") or ""
            output = fn(*args, **kwargs)

            try:
                resp = requests.post(endpoint, json={
                    "input": str(user_input),
                    "output": str(output)
                }).json()

                if not resp.get("allow", True):
                    return {
                        "messages": [
                            {"role": "system", "content": f"❌ Blocked by Governance: {resp['reasons']}"}
                        ]
                    }
                elif resp.get("mode") == "redact":
                    # Replace redactions with ***
                    redacted_output = str(output)
                    for r in resp.get("redactions", []):
                        redacted_output = redacted_output.replace(r, "***")
                    return redacted_output
                else:
                    return output
            except Exception as e:
                print(f"⚠️ Governance check failed: {e}")
                return output
        return wrapper
    return decorator
