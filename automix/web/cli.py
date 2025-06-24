"""
Web interface CLI command
"""
import click

from .app import run_server


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=5000, type=int, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--no-browser", is_flag=True, help="Do not open browser automatically")
def web(host, port, debug, no_browser):
    """Start AutoMix web interface"""
    run_server(host=host, port=port, debug=debug, open_browser=not no_browser)


if __name__ == "__main__":
    web()
