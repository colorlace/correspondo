"""
Data preparation script for Correspondo fine-tuning.
Extracts emails per sender, cleans them, and formats for fine-tuning.
"""

import csv
import json
import quopri
import re
import sys
from pathlib import Path

# Add parent directory to path to import read_emails
sys.path.insert(0, str(Path(__file__).parent.parent))

from read_emails import extract_from_address, extract_latest_reply

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Target personas with their email addresses and display names
PERSONAS = {
    "vince.kaminski@enron.com": {
        "name": "Vince Kaminski",
        "description": "Head of Enron's Research Group, expert in quantitative analysis and risk management"
    },
    "kate.symes@enron.com": {
        "name": "Kate Symes",
        "description": "Enron employee in the trading division"
    },
    "jeff.dasovich@enron.com": {
        "name": "Jeff Dasovich",
        "description": "Enron government affairs representative focused on California energy policy"
    },
    "phillip.allen@enron.com": {
        "name": "Phillip Allen",
        "description": "Enron trader in the gas trading division"
    },
    "enron.announcements@enron.com": {
        "name": "Enron Announcements",
        "description": "Official Enron corporate communications and announcements"
    },
}

MIN_EMAIL_LENGTH = 50  # Minimum characters for a valid email

# Additional patterns to detect quoted/forwarded content that may have leaked through
QUOTE_CLEANUP_PATTERNS = [
    r'\n\s*Enron North America Corp\..*$',
    r'\n\s*From:\s+\S+.*$',
    r'\n\s*Sent by:.*$',
    r'\n\s*To:\s+\S+.*$',
    r'\n\s*cc:\s+\S+.*$',
    r'\n\s*Subject:\s+.*$',
    r'\n\s*@ENRON.*$',
    r'\n\s*<Embedded.*$',
    r'\n={3,}.*$',
    r'\n-{5,}.*$',
]

QUOTE_CLEANUP_RE = re.compile(
    '|'.join(f'({p})' for p in QUOTE_CLEANUP_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL
)


def decode_quoted_printable(text: str) -> str:
    """
    Decode quoted-printable encoded text.
    Handles soft line breaks (=\n) and encoded characters (=XX).
    """
    try:
        # First, handle soft line breaks (= at end of line)
        text = re.sub(r'=\n', '', text)
        # Then decode the quoted-printable encoding
        decoded = quopri.decodestring(text.encode('utf-8', errors='replace'))
        return decoded.decode('utf-8', errors='replace')
    except Exception:
        # If decoding fails, return original with basic cleanup
        text = re.sub(r'=\n', '', text)
        text = re.sub(r'=20', ' ', text)
        text = re.sub(r'=09', '\t', text)
        text = re.sub(r'=([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), text)
        return text


def clean_text(text: str) -> str:
    """
    Apply additional cleaning to email text:
    - Decode quoted-printable
    - Remove leaked quoted content
    - Normalize whitespace
    """
    # Decode quoted-printable encoding
    text = decode_quoted_printable(text)

    # Remove any leaked quoted content
    match = QUOTE_CLEANUP_RE.search(text)
    if match:
        text = text[:match.start()]

    # Normalize multiple blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def extract_emails_by_sender(csv_path: str) -> dict[str, list[str]]:
    """
    Read all emails and group by sender address.
    Returns dict mapping email address -> list of raw email strings.
    """
    sender_emails = {addr: [] for addr in PERSONAS.keys()}

    print(f"Reading emails from {csv_path}...")
    with open(csv_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        total = 0
        for row in reader:
            total += 1
            if total % 100000 == 0:
                print(f"  Processed {total:,} emails...")

            if len(row) > 1:
                addr = extract_from_address(row[1])
                if addr and addr in sender_emails:
                    sender_emails[addr].append(row[1])

    print(f"Total emails processed: {total:,}")
    return sender_emails


def clean_emails(raw_emails: list[str], min_length: int = MIN_EMAIL_LENGTH) -> list[str]:
    """
    Clean emails by extracting only the latest reply, decoding, and filtering short ones.
    """
    cleaned = []
    for email in raw_emails:
        # Extract just the latest reply (no quoted content)
        text = extract_latest_reply(email)
        if not text:
            continue

        # Apply additional cleaning (decode QP, remove leaked quotes, normalize)
        text = clean_text(text)

        # Filter out short/empty emails
        if len(text) >= min_length:
            cleaned.append(text)
    return cleaned


def save_raw_emails(emails: list[str], output_path: Path):
    """Save cleaned emails as JSONL (one email per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for email in emails:
            json.dump({"text": email}, f)
            f.write('\n')


def create_training_data(emails: list[str], persona_info: dict) -> list[dict]:
    """
    Format emails as instruction-tuning examples.
    Uses chat format compatible with most fine-tuning frameworks.
    """
    training_examples = []

    system_prompt = f"You are {persona_info['name']}, {persona_info['description']}. Write in your authentic voice and style."

    for email in emails:
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Write something."},
                {"role": "assistant", "content": email}
            ]
        }
        training_examples.append(example)

    return training_examples


def save_training_data(examples: list[dict], output_path: Path):
    """Save training examples as JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example, f)
            f.write('\n')


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "emails.csv"
    raw_output_dir = project_root / "data" / "raw"
    training_output_dir = project_root / "data" / "training"

    # Ensure output directories exist
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract emails by sender
    sender_emails = extract_emails_by_sender(str(csv_path))

    # Process each persona
    print("\n" + "="*60)
    print("Processing personas...")
    print("="*60)

    all_training_examples = []

    for email_addr, info in PERSONAS.items():
        name_slug = email_addr.split('@')[0].replace('.', '_')
        raw_emails = sender_emails[email_addr]

        print(f"\n{info['name']} ({email_addr})")
        print(f"  Raw emails: {len(raw_emails):,}")

        # Clean emails
        cleaned = clean_emails(raw_emails)
        print(f"  After cleaning (>={MIN_EMAIL_LENGTH} chars): {len(cleaned):,}")

        if not cleaned:
            print(f"  Skipping - no valid emails")
            continue

        # Save raw cleaned emails
        raw_path = raw_output_dir / f"{name_slug}.jsonl"
        save_raw_emails(cleaned, raw_path)
        print(f"  Saved raw: {raw_path}")

        # Create and save training data
        training_data = create_training_data(cleaned, info)
        training_path = training_output_dir / f"{name_slug}.jsonl"
        save_training_data(training_data, training_path)
        print(f"  Saved training: {training_path}")

        all_training_examples.extend(training_data)

    # Save combined training data
    combined_path = training_output_dir / "all_personas.jsonl"
    save_training_data(all_training_examples, combined_path)
    print(f"\nCombined training data: {combined_path}")
    print(f"Total training examples: {len(all_training_examples):,}")


if __name__ == "__main__":
    main()
