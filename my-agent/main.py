import os
import guava
#insurance policy document
from guava.helpers.rag import DocumentQA
from guava.examples.example_data import PROPERTY_INSURANCE_POLICY

document_qa = DocumentQA(documents=PROPERTY_INSURANCE_POLICY)

from typing_extensions import override
from guava import logging_utils

agent = guava.Agent(
    organization_name="Innovative Insurance",
    agent_name="Emma",
    purpose="Answer questions regarding property insurance policy until there are no more questions"
)
agent.listen_phone(os.environ["+14842951565"])

document_qa = DocumentQA(documents=PROPERTY_INSURANCE_POLICY)

@agent.on_call_start()
def on_call_start(call: guava.Call):
    # Set your first task here using call.set_task(...)
    call.set_task()
    pass



if __name__ == "__main__":
    logging_utils.configure_logging()
    agent.listen_phone(os.environ["GUAVA_AGENT_NUMBER"])


# ADD CODE