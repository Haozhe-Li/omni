from core.supervisors import *
from core.utils import pretty_print_messages

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "请帮我写一个binary search",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)
