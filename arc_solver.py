"""Deprecated entrypoint.

Use `arc_agi3_pmll_agent.py` for ARC-AGI-3 interactive runs.
"""

import asyncio

from arc_agi3_pmll_agent import main


if __name__ == "__main__":
    asyncio.run(main())
