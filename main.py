import guava

from typing_extensions import override
from guava import logging_utils

agent = guava.Agent(
    purpose="You are a helpful voice agent."
)


@agent.on_call_start()
def on_call_start(call: guava.Call):
    # Set your first task here using call.set_task(...)
    pass


if __name__ == "__main__":
    logging_utils.configure_logging()
    agent.listen_phone("+14842950149")