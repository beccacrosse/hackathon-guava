import argparse
import csv
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import guava
from guava import logging_utils

COMPLAINT_DATABASE_CSV = Path("insurance_complaints.csv")
DETAILED_CASES_CSV = Path("cases.csv")
DETAILED_CASE_FIELDS = [
    "timestamp",
    "full_name",
    "date_of_birth",
    "phone_number",
    "current_client",
    "insurance_number",
    "support_sector",
    "case_number",
    "inquiry_summary",
    "problem_characteristics",
    "multiple_attempts",
    "inquiry_solved",
]

agent = guava.Agent(
    name="Emma",
    organization="Your customer service representative",
    purpose=(
        "to answer inbound insurance support calls, gather complaint details, "
        "classify the issue, and document whether it can be resolved or needs human follow-up"
    ),
)


def normalize_text(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def compute_similarity(text_a: str, text_b: str) -> float:
    tokens_a = normalize_text(text_a)
    tokens_b = normalize_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union)


def load_complaint_database(path: Path = COMPLAINT_DATABASE_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader if row.get("Complaint")]


def append_complaint_case(row: dict[str, str], path: Path = COMPLAINT_DATABASE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["Complaint", "Representative Response & Resolution Steps"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_detailed_case(results: dict[str, str], path: Path = DETAILED_CASES_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=DETAILED_CASE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(results)


def load_detailed_cases(path: Path = DETAILED_CASES_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def normalize_phone_number(phone_number: str) -> str:
    return re.sub(r"\D", "", phone_number or "")


def find_returning_caller_case(
    phone_number: str, date_of_birth: str, cases: list[dict[str, str]]
) -> dict[str, str] | None:
    normalized_phone = normalize_phone_number(phone_number)
    normalized_dob = (date_of_birth or "").strip().lower()
    if not normalized_phone or not normalized_dob:
        return None

    for case in reversed(cases):
        case_phone = normalize_phone_number(case.get("phone_number", ""))
        case_dob = (case.get("date_of_birth", "") or "").strip().lower()
        if case_phone == normalized_phone and case_dob == normalized_dob:
            return case
    return None


def find_similar_case(query: str, cases: list[dict[str, str]], threshold: float = 0.3) -> dict[str, str] | None:
    best_match = None
    best_score = 0.0
    for case in cases:
        score = compute_similarity(query, case.get("Complaint", ""))
        if score > best_score:
            best_match = case
            best_score = score
    if best_score >= threshold:
        logging.info("Found similar complaint case with score %.2f", best_score)
        return best_match
    return None


@agent.on_call_start
def on_call_start(call: guava.Call) -> None:
    call.set_task(
        "insurance_complaint intake",
        objective=(
            "You are answering an inbound complaint call for ABC Insurance Company. "
            "Greet the caller warmly as Emma, confirm their identity, gather their customer and policy details, "
            "understand the nature of their complaint, classify the support sector, and determine whether the issue "
            "has already been solved or requires a human callback. Keep your tone empathetic and conversational, "
            "and vary acknowledgment language naturally rather than repeating the same phrase. Rotate responses like "
            "'Thanks for sharing that', 'I hear you', 'That sounds frustrating', 'I appreciate you explaining that', "
            "'Got it', and 'Thanks for your patience'. Only use empathy acknowledgments when the caller has shared a "
            "concern or issue; for simple greetings like 'hello', respond with a natural greeting instead of saying "
            "'I understand'. If the caller's request is unclear or ambiguous, explicitly ask whether they can provide "
            "any additional information before continuing. Do NOT ask for the insurance or policy number if the "
            "caller says they are not a current client. Do not say you are transferring to a specialist. Before "
            "ending the call, always ask whether the caller needs any additional help, then close with a pleasant "
            "goodbye. Do not ask the same question multiple times: once a field is answered, briefly acknowledge it "
            "and move to the next required item."
        ),
        checklist=[
            guava.Say(
                "Hello, this is Emma, your customer service representative for ABC Insurance Company. "
                "I am here to help with your insurance question or complaint."
            ),
            guava.Field(
                key="full_name",
                description="The caller's full name",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="date_of_birth",
                description="The caller's date of birth",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="phone_number",
                description="The caller's callback phone number",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="current_client",
                description="Whether the caller is a current client (yes/no)",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="insurance_number",
                description="The caller's insurance or policy number if the caller has one",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="support_sector",
                description=(
                    "The support area that best fits the inquiry, such as billing, coverage, claims, "
                    "or enrollment"
                ),
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="case_number",
                description="A case number if the caller already has one",
                field_type="text",
                required=False,
            ),
            guava.Field(
                key="inquiry_summary",
                description="A short description of the caller's question or complaint",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="problem_characteristics",
                description=(
                    "Key facts about the problem, including what happened, when it began, "
                    "and any impacts to the caller"
                ),
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="multiple_attempts",
                description="Whether the caller has already made multiple attempts to resolve the issue (yes/no)",
                field_type="text",
                required=True,
            ),
            guava.Field(
                key="inquiry_solved",
                description="Whether the caller's issue has already been solved (yes/no)",
                field_type="text",
                required=True,
            ),
        ],
    )


@agent.on_task_complete("insurance_complaint intake")
def on_complaint_complete(call: guava.Call) -> None:
    details = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "full_name": call.get_field("full_name"),
        "date_of_birth": call.get_field("date_of_birth"),
        "phone_number": call.get_field("phone_number"),
        "current_client": call.get_field("current_client"),
        "insurance_number": call.get_field("insurance_number"),
        "support_sector": call.get_field("support_sector"),
        "case_number": call.get_field("case_number") or "",
        "inquiry_summary": call.get_field("inquiry_summary"),
        "problem_characteristics": call.get_field("problem_characteristics"),
        "multiple_attempts": call.get_field("multiple_attempts"),
        "inquiry_solved": call.get_field("inquiry_solved"),
    }

    search_text = (
        f"{details['inquiry_summary']} {details['problem_characteristics']} {details['support_sector']}"
    )
    detailed_cases = load_detailed_cases()
    returning_case = find_returning_caller_case(
        details["phone_number"], details["date_of_birth"], detailed_cases
    )
    past_cases = load_complaint_database()
    match = find_similar_case(search_text, past_cases)

    if returning_case is not None:
        logging.info(
            "Returning caller detected for phone %s and DOB %s",
            details["phone_number"],
            details["date_of_birth"],
        )
        if not details["case_number"] and returning_case.get("case_number"):
            details["case_number"] = returning_case["case_number"]

    if match is not None:
        resolution_steps = match["Representative Response & Resolution Steps"]
        if resolution_steps is not None:
            final_instructions = (
                "I verified your information and found your prior case details on file. "
                "I also found a similar resolved complaint, so your issue is already underway using the established "
                "resolution process. Your case has been escalated, and a human representative will call you back "
                "with an update. If anything changes, we will follow up with you. Before we end, is there anything "
                "else I can help you with today? Thank you for calling ABC Insurance, and have a great day."
            )
        else:
            final_instructions = (
                "I found a similar case in our records. I have restated and recorded your issue, and it is now "
                "underway using the same resolution approach that worked before. Your case has been escalated, and "
                "a human representative will call you back with an update. Thank you for your patience. Before we "
                "end, is there anything else I can help you with today? Thank you for calling ABC Insurance, and "
                "have a great day."
            )
    else:
        if returning_case is not None:
            final_instructions = (
                "I verified your information and confirmed your previous details are already on file. "
                "I have added today's update, and a human representative will review your case. "
                "Please expect a callback within 5 business days. Before we end, is there anything else I can help "
                "you with today? Thank you for calling ABC Insurance, and have a great day."
            )
        else:
            final_instructions = (
                "Thank you for explaining the issue. I was unable to find a similar case in our records, "
                "so a human representative will personally review your case and call you back. "
                "Please expect a callback within 5 business days. Before we end, is there anything else I can help "
                "you with today? Thank you for calling ABC Insurance, and have a great day."
            )

    append_detailed_case(details)
    append_complaint_case(
        {
            "Complaint": f"{details['inquiry_summary']} {details['problem_characteristics']}",
            "Representative Response & Resolution Steps": resolution_steps,
        }
    )

    logging.info("Updated complaint database at %s", COMPLAINT_DATABASE_CSV)
    logging.info("Saved detailed case record to %s", DETAILED_CASES_CSV)

    call.hangup(final_instructions=final_instructions)


if __name__ == "__main__":
    logging_utils.configure_logging()
    parser = argparse.ArgumentParser(
        description="Inbound insurance complaints voice agent for ABC Insurance Company"
    )
    parser.add_argument(
        "--agent-number",
        default="+14844813864",
        help="Phone number the agent should listen on for inbound calls",
    )
    args = parser.parse_args()

    if not args.agent_number:
        raise SystemExit(
            "Missing agent number. Set GUAVA_AGENT_NUMBER or pass --agent-number."
        )

    agent.listen_phone(args.agent_number)
