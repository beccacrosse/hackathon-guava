import logging
import os
import guava
import argparse

from guava.helpers.rag import DocumentQA
from guava import logging_utils, Agent
from guava.examples.example_data import PROPERTY_INSURANCE_POLICY

agent = guava.Agent(
    organization_name="Innovative Insurance",
    agent_name="Emma",
    purpose="Answer questions regarding property insurance policy until there are no more questions"
)
agent.listen_phone("+14844813864")

document_qa = DocumentQA(documents=PROPERTY_INSURANCE_POLICY)

logger = logging.getLogger("guava.examples.property_insurance")

@agent.on_call_start()
def on_call_start(call: guava.Call):
    # Set your first task here using call.set_task(...)
    call.set_task("say hello to the caller, ask them how you can help them today")
    pass
# When the Agent is asked a question that it cannot answer, it will invoke the on_question callback.
@agent.on_question
def on_question(call: guava.Call, question: str) -> str:
    # Forward the Agent's question to the knowledge base and return the answer.
    # You can plug in any knowledge base system you want here.
    answer = document_qa.ask(question)
    logger.info("RAG answer: %s", answer)
    return answer


if __name__ == "__main__":
    logging_utils.configure_logging()

    # Every Agent can be attached to multiple resources.
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phone", action="store_true", help="Listen for phone calls.")
    group.add_argument("--webrtc", action="store_true", help="Create on a WebRTC code.")
    group.add_argument("--local", action="store_true", help="Start a local call.")
    args = parser.parse_args()

    # We can attach our agent to receive inbound phone or WebRTC calls.
    if args.phone:
        agent.listen_phone(os.environ["GUAVA_AGENT_NUMBER"])
    elif args.webrtc:
        agent.listen_webrtc()
    else:
        agent.local_call()



# if __name__ == "__main__":
#     logging_utils.configure_logging()
#     agent.listen_phone(os.environ["GUAVA_AGENT_NUMBER"])


# # ADD CODE
# import os
# import guava
# #insurance policy document
# from guava.helpers.rag import DocumentQA
# from guava.examples.example_data import PROPERTY_INSURANCE_POLICY

# document_qa = DocumentQA(documents=PROPERTY_INSURANCE_POLICY)

# from typing_extensions import override
# from guava import logging_utils
