from core.supervisors import *
from core.utils import pretty_print_messages, pretty_print_message

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
    pretty_print_messages(chunk, last_message=True)
