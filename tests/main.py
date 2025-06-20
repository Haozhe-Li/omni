from core.supervisors import *
from core.utils import pretty_yield_messages

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请给我介绍一下成都",
            }
        ]
    },
):
    pretty_yield_messages(chunk, last_message=True)
