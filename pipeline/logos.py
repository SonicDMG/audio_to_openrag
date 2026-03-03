"""
Logo display utilities for CLI output.

Provides ASCII art representations of service logos to display
contextually throughout the pipeline execution.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


# ASCII art logos
DOCLING_LOGO = r"""
    ____             ___
   / __ \____  _____/ (_)___  ____ _
  / / / / __ \/ ___/ / / __ \/ __ `/
 / /_/ / /_/ / /__/ / / / / / /_/ /
/_____/\____/\___/_/_/_/ /_/\__, /
                           /____/
"""

OPENRAG_LOGO = r"""
   ____                   ____  ___   ______
  / __ \____  ___  ____  / __ \/   | / ____/
 / / / / __ \/ _ \/ __ \/ /_/ / /| |/ / __
/ /_/ / /_/ /  __/ / / / _, _/ ___ / /_/ /
\____/ .___/\___/_/ /_/_/ |_/_/  |_\____/
    /_/
"""

YOUTUBE_LOGO = r"""
 __   __          _____      _
 \ \ / /__  _   _|_   _|   _| |__   ___
  \ V / _ \| | | | | || | | | '_ \ / _ \
   | | (_) | |_| | | || |_| | |_) |  __/
   |_|\___/ \__,_| |_| \__,_|_.__/ \___|
"""


def display_logo(console: Console, logo_type: str, message: str = "") -> None:
    """Display a logo with an optional message.
    
    Args:
        console: Rich console instance for output
        logo_type: Type of logo to display ('docling', 'openrag', 'youtube')
        message: Optional message to display below the logo
    """
    logo_map = {
        "docling": (DOCLING_LOGO, "cyan"),
        "openrag": (OPENRAG_LOGO, "green"),
        "youtube": (YOUTUBE_LOGO, "red"),
    }

    if logo_type.lower() not in logo_map:
        return

    logo_text, color = logo_map[logo_type.lower()]

    # Create styled text
    styled_logo = Text(logo_text, style=f"bold {color}")

    if message:
        styled_logo.append(f"\n{message}", style=f"{color}")

    # Display in a panel
    panel = Panel(
        styled_logo,
        border_style=color,
        padding=(0, 2),
    )
    console.print(panel)


# Made with Bob
