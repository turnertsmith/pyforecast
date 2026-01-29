"""CLI entry point for PyForecast."""

from pyforecast.cli.commands import app


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
