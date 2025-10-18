"""Entrypoint for the project. Runs the FastAPI app for local development.

This file keeps a minimal role: when executed as a script it starts the ASGI
server (uvicorn) pointing to `src.app:app`. Keeping the app itself in
`src/app.py` makes it easier to import the FastAPI app in tests or other
modules without launching the server.
"""

# Run: uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

if __name__ == "__main__":
	import uvicorn

	# Recommended for development. In production use a proper ASGI server config.
	uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)