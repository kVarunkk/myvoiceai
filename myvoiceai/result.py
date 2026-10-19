from dataclasses import dataclass, field
from typing import Literal

EndReason = Literal["completed", "time_limit", "inactivity", "client_disconnect", "error"]


@dataclass
class SessionResult:
    """What a voice session reports when it ends.

    `turns` holds each turn record exactly as it was sent to the client
    (`turn_id`, `timestamp`, `user`, `assistant`, `interrupted`). Times are
    Unix timestamps in seconds.
    """

    turns: list[dict] = field(default_factory=list)
    end_reason: EndReason = "error"
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_seconds: float = 0.0
