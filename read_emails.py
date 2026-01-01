"""
This script reads the first 5 emails from emails.csv and puts them into a list of lists.
"""

import csv
import re
import sys

# Increase CSV field size limit to handle large emails
csv.field_size_limit(sys.maxsize)
import re
from typing import Optional

# Common “start of quoted history” markers (case-insensitive)
_QUOTE_CUTOFF_PATTERNS = [
    r"^-{2,}\s*original message\s*-{2,}\s*$",
    r"^-{2,}\s*forwarded by.*$",
    r"^-{2,}\s*forwarded message\s*-{2,}\s*$",
    r"^from:\s.*$",          # embedded header block
    r"^to:\s.*$",
    r"^cc:\s.*$",
    r"^bcc:\s.*$",
    r"^subject:\s.*$",
    r"^date:\s.*$",
    r"^sent:\s.*$",
    r"^on .+ wrote:\s*$",    # "On ... wrote:"
]

_CUTOFF_RE = re.compile("|".join(f"({p})" for p in _QUOTE_CUTOFF_PATTERNS),
                        flags=re.IGNORECASE | re.MULTILINE)

def extract_latest_reply(email_text: str) -> str:
    """
    Extract only the latest reply text from a raw Enron email string.

    Steps:
      1) Split headers from body by the first blank line.
      2) In the body, cut off at the first sign of quoted history / embedded headers.
    """
    # 1) Header/body split (Enron format is usually reliable here)
    parts = email_text.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else ""
    body = body.strip()

    if not body:
        return ""

    # 2) Cut off quoted history
    m = _CUTOFF_RE.search(body)
    if m:
        body = body[:m.start()].rstrip()

    return body

def drop_email_metadata(email: str) -> str:
	"""Takes an email string and returns just the body, removing all metadata headers."""
	lines = email.split('\n')
	
	# Find the first blank line that separates headers from body
	body_start = 0
	for i, line in enumerate(lines):
		if line.strip() == '':
			body_start = i + 1
			break
	
	# Return the body, stripping leading/trailing whitespace
	return '\n'.join(lines[body_start:]).strip()


def read_emails(file_path, num_emails=None) -> list[list[str]]:
	emails = []
	with open(file_path, mode='r', newline='', encoding='utf-8') as file:
		reader = csv.reader(file)
		if num_emails is None:
			emails = list(reader)
		else:
			for i, row in enumerate(reader):
				if i < num_emails:
					emails.append(row)
				else:
					break
	return emails



def is_from(email_content: str, address: str) -> bool:
	"""Checks if the email is from the specified address."""
	return f"From: {address}" in email_content


def is_from_enron_announcements(email_content: str) -> bool:
	"""Checks if the email is from enron.announcements@enron.com"""
	email_address = "enron.announcements@enron.com"
	return is_from(email_content, email_address)


def is_from_kate_symes(email_content: str) -> bool:
	return is_from(email_content, "kate.symes@enron.com")


def is_from_vince_kaminski(email_content: str) -> bool:
	return is_from(email_content, "vince.kaminski@enron.com")


def extract_from_address(email_content: str) -> str | None:
	"""Extracts the From email address from an email string."""
	match = re.search(r'^From:\s*([^\s@]+@[^\s@]+\.[^\s@]+)', email_content, re.MULTILINE)
	return match.group(1) if match else None


def get_all_from_addresses(file_path: str) -> list[str]:
	"""Reads all emails and returns a list of unique From addresses."""
	from_addresses = []
	with open(file_path, mode='r', newline='', encoding='utf-8') as file:
		reader = csv.reader(file)
		for row in reader:
			if len(row) > 1:
				addr = extract_from_address(row[1])
				if addr:
					from_addresses.append(addr)
	return from_addresses


if __name__ == "__main__":
	"""collect a list of vince kaminski and enron announcement emails"""
	email_list = read_emails('emails.csv')

	# create a dictionary of sender address to list of emails from that sender
	sender_dict = {}
	for email in email_list:
		addr = extract_from_address(email[1])
		if addr:
			if addr not in sender_dict:
				sender_dict[addr] = []
			sender_dict[addr].append(email[1])
	print(f"Total unique senders: {len(sender_dict)}")
	# print the number of emails from vince kaminski
	print(f"Number of emails from vince.kaminski@enron.com: {len(sender_dict['vince.kaminski@enron.com'])}")
	print(f"Number of emails from enron.announcements@enron.com: {len(sender_dict['enron.announcements@enron.com'])}")
	print(f"Number of emails from phillip.allen@enron.com: {len(sender_dict['phillip.allen@enron.com'])}")
	print(f"Number of emails from Kate Symes: {len(sender_dict['kate.symes@enron.com'])}")
	print(f"Number of emails from jeff.dasovich@enron.com: {len(sender_dict['jeff.dasovich@enron.com'])}")

	print("Top 5 senders by number of emails:")
	# sort the dictionary by number of emails and print the top 5
	sorted_senders = sorted(sender_dict.items(), key=lambda item: len(item[1]), reverse=True)
	for sender, emails in sorted_senders[:5]:
		print(f"• {sender}: {len(emails)} emails")

    # Example usage of extract_latest_reply
	for email in email_list[:15]:
		latest_reply = extract_latest_reply(email[1])
		print(f"{email[0]}:\n{latest_reply}\n{'-'*40}\n")