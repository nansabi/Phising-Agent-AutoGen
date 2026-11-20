"""Offline tools for extracting email facts and scoring links and domains."""

import email.policy
from email.parser import Parser
from email.utils import parseaddr
import ipaddress
import re
from urllib.parse import urlsplit

from pydantic import BaseModel


class LinkInput(BaseModel):
    """Input passed to the link risk checker."""

    url: str


class LinkResult(BaseModel):
    """A URL, its rule-based risk score, and the matching indicators."""

    url: str
    risk_score: int
    reasons: list[str]


class DomainResult(BaseModel):
    """A sender domain, its rule-based risk score, and matching indicators."""

    domain: str
    risk_score: int
    reasons: list[str]


def extract_links(email_text: str) -> list[str]:
    """Return HTTP(S) or www links found in the email text."""
    matches = re.findall(r"""https?://[^\s<>"']+|www\.[^\s<>"']+""", email_text, re.IGNORECASE)
    return [match.rstrip(".,;:!?)]}") for match in matches]


def extract_sender(email_text: str) -> str:
    """Return the From header value, or a clear placeholder if it is absent."""
    match = re.search(r"(?im)^From:\s*(.+?)\s*$", email_text)
    return match.group(1).strip() if match else "Not found"


def extract_headers(email_text: str) -> dict[str, str | list[str]]:
    """Return selected visible email headers, preserving repeated Received fields."""
    parsed = Parser(policy=email.policy.default).parsestr(
        email_text,
        headersonly=True,
    )
    wanted = (
        "From",
        "Reply-To",
        "Return-Path",
        "Received",
        "Authentication-Results",
    )
    headers: dict[str, str | list[str]] = {}
    for name in wanted:
        values = parsed.get_all(name, [])
        if values:
            rendered = [str(value) for value in values]
            headers[name] = rendered if name == "Received" else rendered[-1]
    return headers


def extract_keywords(email_text: str) -> list[str]:
    """Find a concise set of common urgency and credential-related terms."""
    keywords = (
        "verify",
        "suspended",
        "urgent",
        "immediately",
        "password",
        "account",
        "confirm",
        "payment",
        "login",
        "security alert",
    )
    text = email_text.casefold()
    return [word for word in keywords if re.search(rf"\b{re.escape(word)}\b", text)]


def check_sender_domain(
    from_header: str,
    reply_to: str | None,
) -> DomainResult:
    """Score common sender-domain mismatches without contacting any service."""
    display_name, sender_address = parseaddr(from_header)
    sender_domain = _email_domain(sender_address)
    reply_domain = _email_domain(parseaddr(reply_to or "")[1])
    domain = sender_domain or "unknown"
    score = 0
    reasons: list[str] = []

    brand_domains = {
        "amazon": "amazon.com",
        "apple": "apple.com",
        "google": "google.com",
        "microsoft": "microsoft.com",
        "paypal": "paypal.com",
    }
    display_text = display_name.casefold()
    claimed_brands = [
        brand for brand in brand_domains if brand in display_text
    ]
    for brand in claimed_brands:
        expected_domain = brand_domains[brand]
        if not _is_domain_or_subdomain(sender_domain, expected_domain):
            score += 40
            reasons.append(
                f"Display name claims {brand.title()} but sender domain is "
                f"{sender_domain or 'unknown'}"
            )

    if reply_domain and sender_domain and reply_domain != sender_domain:
        score += 35
        reasons.append(
            f"Reply-To domain {reply_domain} differs from From domain "
            f"{sender_domain}"
        )

    typo_markers = ("paypa1", "micros0ft", "rnicrosoft")
    matched_markers = [word for word in typo_markers if word in sender_domain]
    if matched_markers:
        score += 35
        reasons.append(
            "Sender domain contains a lookalike spelling: "
            + ", ".join(matched_markers)
        )

    if sender_domain.rsplit(".", 1)[-1] in {"xyz", "top", "click"}:
        score += 25
        reasons.append("Sender domain uses a suspicious top-level domain")

    labels = sender_domain.split(".") if sender_domain else []
    hyphen_count = sender_domain.count("-")
    if hyphen_count >= 2 or len(labels) >= 5:
        score += 15
        reasons.append("Sender domain has many hyphens or subdomains")

    free_mail_domains = {
        "gmail.com",
        "googlemail.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "yahoo.com",
        "icloud.com",
        "aol.com",
    }
    company_claim = bool(claimed_brands) or any(
        word in display_text
        for word in ("support", "billing", "security", "customer service")
    )
    if company_claim and sender_domain in free_mail_domains:
        score += 25
        reasons.append("Company display name uses a free-mail sender domain")

    if not reasons:
        reasons.append("No configured sender-domain risk indicators were found")

    return DomainResult(
        domain=domain,
        risk_score=min(score, 100),
        reasons=reasons,
    )


def check_auth_results(headers: dict[str, str | list[str]]) -> list[str]:
    """Report visible SPF, DKIM, and DMARC outcomes from Authentication-Results."""
    raw_results = headers.get("Authentication-Results")
    if isinstance(raw_results, list):
        auth_text = " ".join(raw_results)
    elif raw_results:
        auth_text = raw_results
    else:
        auth_text = ""

    outcomes: list[str] = []
    for protocol in ("SPF", "DKIM", "DMARC"):
        match = re.search(
            rf"\b{protocol}\s*=\s*(pass|fail|softfail|neutral|none|"
            r"temperror|permerror)\b",
            auth_text,
            flags=re.IGNORECASE,
        )
        status = match.group(1).casefold() if match else "not available"
        outcomes.append(f"{protocol}: {status}")
    return outcomes


def compare_link_domains(sender_domain: str, links: list[str]) -> list[str]:
    """List links whose apparent domain differs from the sender's base domain."""
    sender_base = _base_domain(sender_domain.casefold().strip("."))
    findings: list[str] = []
    for link in links:
        normalized_link = link if "://" in link else f"https://{link}"
        link_domain = (urlsplit(normalized_link).hostname or "").casefold().strip(".")
        if not link_domain:
            findings.append(f"Could not determine the domain for link: {link}")
            continue
        if _base_domain(link_domain) != sender_base:
            findings.append(
                f"Link domain {link_domain} is unrelated to sender domain "
                f"{sender_domain or 'unknown'}"
            )
    if not findings and links:
        findings.append("No unrelated link domains were found")
    elif not links:
        findings.append("No links found")
    return findings


def check_link(input: LinkInput) -> LinkResult:
    """Score common suspicious URL characteristics without visiting the URL."""
    url = input.url.strip()
    normalized_url = url if "://" in url else f"https://{url}"
    parsed = urlsplit(normalized_url)
    hostname = (parsed.hostname or "").casefold().rstrip(".")

    score = 0
    reasons: list[str] = []

    try:
        ipaddress.ip_address(hostname)
        score += 40
        reasons.append("URL uses an IP address instead of a domain name")
    except ValueError:
        pass

    if parsed.scheme.casefold() != "https":
        score += 20
        reasons.append("URL does not use HTTPS")

    if len(url) > 100:
        score += 15
        reasons.append("URL is unusually long")

    if hostname.rsplit(".", 1)[-1] in {"xyz", "top"}:
        score += 25
        reasons.append("URL uses a commonly abused top-level domain")

    if "@" in url:
        score += 35
        reasons.append("URL contains an @ character that can obscure its destination")

    lookalike_words = ("paypa1", "micros0ft")
    matched_lookalikes = [word for word in lookalike_words if word in hostname]
    if matched_lookalikes:
        score += 35
        reasons.append(
            "Hostname contains a lookalike brand spelling: "
            + ", ".join(matched_lookalikes)
        )

    shorteners = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "cutt.ly",
        "shorturl.at",
        "rebrand.ly",
    }
    if hostname in shorteners:
        score += 20
        reasons.append("URL uses a link-shortening service")

    if not reasons:
        reasons.append("No configured URL risk indicators were found")

    return LinkResult(url=url, risk_score=min(score, 100), reasons=reasons)


def check_link_for_agent(url: str) -> str:
    """Expose the Pydantic-backed checker through a simple AutoGen tool schema."""
    result = check_link(LinkInput(url=url))
    return result.json()


def check_sender_domain_for_agent(
    from_header: str,
    reply_to: str | None = None,
) -> str:
    """Return a JSON representation of the sender-domain assessment."""
    result = check_sender_domain(from_header, reply_to)
    return result.json()


def _email_domain(address: str) -> str:
    """Extract and normalize the domain portion of an email address."""
    if "@" not in address:
        return ""
    return address.rsplit("@", 1)[1].casefold().strip(".")


def _is_domain_or_subdomain(domain: str, expected: str) -> bool:
    """Check a domain boundary rather than a misleading substring match."""
    return domain == expected or domain.endswith(f".{expected}")


def _base_domain(domain: str) -> str:
    """Use the final two labels as an offline, approximate domain boundary."""
    labels = [label for label in domain.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else domain
