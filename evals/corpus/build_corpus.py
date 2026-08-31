"""Regenerate the synthetic and typeset PDFs in this directory.

Three of the four corpus documents are produced here so the exact bytes
the eval runs against are reproducible from text held in the repo:

    notes_primer.pdf       synthetic  -- a CS networking study primer,
                                          one topic per page, dense
                                          informal prose (the "study
                                          notes" shape the original
                                          scratch/sample.pdf had)
    thermostat_manual.pdf  synthetic  -- a fictional product manual,
                                          imperative voice, spec and
                                          troubleshooting tables
    apache_license_2.0.pdf typeset    -- the verbatim Apache License 2.0
                                          text (apache_license_2.0.txt in
                                          this dir), laid out one clause
                                          group per page

The fourth, nist_sp800-63-3.pdf, is a real published PDF committed
as-is -- see SOURCES.md. This script does not touch it.

Deliberately NOT wired into pytest and NOT called by the eval scripts at
run time: the committed PDFs are the source of truth for the README's
numbers, and regenerating them silently on every eval run would let a
stray edit here move those numbers without anyone noticing. Run it by
hand when you actually mean to change the corpus, then commit the PDFs.

Usage:
    python evals/corpus/build_corpus.py
"""

from __future__ import annotations

import pathlib

import fitz

HERE = pathlib.Path(__file__).parent

# --- page geometry -------------------------------------------------------
# US Letter, generous margins. Body text is inserted into TEXT_RECT; if a
# page's text ever overflows it, insert_textbox() returns a negative
# number and build() raises rather than silently clipping.
PAGE_W, PAGE_H = 612, 792
MARGIN = 72
HEAD_FONTSIZE = 15
BODY_FONTSIZE = 10.5
BODY_LEADING = 15
TEXT_RECT = fitz.Rect(MARGIN, MARGIN + 34, PAGE_W - MARGIN, PAGE_H - MARGIN)


def _write_pages(path: pathlib.Path, title: str, pages: list[tuple[str, str]]) -> None:
    """pages is [(page_heading, page_body), ...] -- one PDF page each."""
    doc = fitz.open()
    for heading, body in pages:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text(
            (MARGIN, MARGIN + 6), heading, fontsize=HEAD_FONTSIZE, fontname="hebo"
        )
        page.draw_line(
            (MARGIN, MARGIN + 16),
            (PAGE_W - MARGIN, MARGIN + 16),
            width=0.6,
        )
        leftover = page.insert_textbox(
            TEXT_RECT,
            body.strip(),
            fontsize=BODY_FONTSIZE,
            fontname="helv",
            lineheight=BODY_LEADING / BODY_FONTSIZE,
        )
        if leftover < 0:
            raise ValueError(
                f"{path.name}: page {doc.page_count} ({heading!r}) overflows its "
                f"text box by ~{-leftover:.0f}pt -- shorten the copy or split the page."
            )
    # Pin every timestamp so regenerating the corpus produces byte-identical
    # PDFs (PyMuPDF otherwise stamps the current time into the metadata and
    # the trailer /ID, which would show up as a spurious git diff).
    _FIXED = "D:20240101000000Z"
    doc.set_metadata(
        {"title": title, "creationDate": _FIXED, "modDate": _FIXED, "producer": "", "creator": ""}
    )
    doc.xref_set_key(-1, "ID", "[<00><00>]")
    doc.save(str(path), garbage=4, deflate=True)
    n = doc.page_count
    doc.close()
    print(f"wrote {path.relative_to(HERE.parent.parent)}  ({n} pages)")


# ======================================================================
# 1. notes_primer.pdf  -- synthetic CS study notes
# ======================================================================
NOTES_TITLE = "Computer Networking: A Study Primer"
NOTES_PAGES: list[tuple[str, str]] = [
    (
        "1. Protocols and the layered model",
        "A protocol is an agreed set of rules for formatting and exchanging messages "
        "between machines. Real networks stack several protocols on top of one another, "
        "each solving one part of the problem. This primer uses the four-layer TCP/IP "
        "model: the link layer moves frames between two directly connected machines; "
        "the internet layer moves packets across networks using IP addresses; the "
        "transport layer gives applications either a reliable ordered byte stream (TCP) "
        "or lightweight best-effort datagrams (UDP); and the application layer is where "
        "protocols such as HTTP and DNS live.\n\n"
        "Each layer wraps the data handed down from the layer above it in its own "
        "header, a process called encapsulation. As a packet travels down the sending "
        "stack it gains a transport header, then an IP header, then a link-layer frame "
        "header; the receiving stack strips those headers off in the reverse order on "
        "the way back up. A layer only ever reads its own header, which is what lets "
        "one layer be changed without disturbing the others.",
    ),
    (
        "2. The link layer",
        "The link layer delivers frames between machines on the same physical or "
        "logical network segment. Every network interface has a MAC address: a 48-bit "
        "identifier, usually written as six hex bytes, that is assigned by the "
        "manufacturer and is meant to be globally unique.\n\n"
        "An Ethernet frame carries a destination MAC, a source MAC, a type field, the "
        "payload handed down from the internet layer, and a trailing checksum. A switch "
        "forwards frames by learning which MAC address sits on which port: it inspects "
        "the source address of every frame it receives, records the port it arrived on, "
        "and from then on sends frames for that destination out only that one port. "
        "Frames for an address it has not yet learned, and broadcast frames, are "
        "flooded out every port except the one they came in on. All the interfaces a "
        "broadcast frame can reach form one broadcast domain.",
    ),
    (
        "3. The internet layer and IP addressing",
        "The internet layer moves packets between networks. An IPv4 address is 32 bits, "
        "written as four decimal octets separated by dots, for example 203.0.113.7. "
        "Three ranges are reserved for private use and are never routed on the public "
        "internet: 10.0.0.0/8, 172.16.0.0/12, and 192.168.0.0/16. Hosts on a private "
        "network reach the internet through network address translation at their "
        "gateway.\n\n"
        "IPv4's roughly four billion addresses proved too few, so IPv6 was introduced "
        "with 128-bit addresses, written as eight groups of four hex digits with runs "
        "of zeros collapsed to a double colon. IPv6 also removes the need for NAT in "
        "most deployments because every host can have a globally unique address.",
    ),
    (
        "4. Subnets and CIDR notation",
        "A subnet is a contiguous block of IP addresses sharing a common prefix. CIDR "
        "notation writes the prefix length after a slash: in 192.168.1.0/24 the first "
        "24 bits are the network portion and the remaining 8 bits identify a host "
        "within it.\n\n"
        "A /24 therefore spans 256 addresses, but two are not usable as host "
        "addresses: the all-zeros host part is the network address that names the "
        "subnet itself, and the all-ones host part is the broadcast address for that "
        "subnet. That leaves 254 usable host addresses in a /24. Every bit added to the "
        "prefix halves the block: a /25 is 128 addresses, a /26 is 64, and so on. "
        "Routers and hosts decide whether a destination is 'on my subnet' or 'must go "
        "through the router' by comparing prefixes.",
    ),
    (
        "5. Routing and the path across the internet",
        "A router connects two or more networks and forwards packets between them. It "
        "keeps a routing table of destination prefixes, each paired with a next hop. "
        "When a packet arrives the router picks the table entry whose prefix matches "
        "the destination address and is the most specific, that is, has the longest "
        "prefix length. This is longest-prefix match. If nothing else matches, the "
        "default route 0.0.0.0/0 is used; on a host that next hop is the default "
        "gateway.\n\n"
        "Every IP packet carries a time-to-live field. Each router that forwards the "
        "packet decrements the TTL by one, and if the TTL reaches zero the packet is "
        "discarded and the router sends an ICMP time-exceeded message back to the "
        "source. This both stops packets from circulating forever in a routing loop "
        "and is the mechanism the traceroute tool exploits to map a path hop by hop.",
    ),
    (
        "6. The transport layer",
        "Two transport protocols sit on top of IP. TCP provides a connection-oriented, "
        "reliable, ordered byte stream: bytes handed to the sender arrive once, in "
        "order, or the connection fails. UDP provides connectionless best-effort "
        "datagrams with no delivery, ordering, or duplicate-suppression guarantees; it "
        "is used where low overhead matters more than reliability, such as DNS queries "
        "and live media.\n\n"
        "Both use 16-bit port numbers to identify which application a segment belongs "
        "to, so a single host can hold many connections at once. Ports below 1024 are "
        "the well-known ports reserved for standard services. A TCP connection opens "
        "with a three-way handshake: the client sends a SYN, the server replies with a "
        "SYN-ACK, and the client answers with an ACK, after which data can flow.",
    ),
    (
        "7. How TCP stays reliable",
        "TCP numbers every byte it sends. Each segment carries a sequence number "
        "identifying its first byte, and each acknowledgement carries the number of the "
        "next byte the receiver expects, which implicitly confirms everything before "
        "it. If an acknowledgement does not arrive before a retransmission timer "
        "expires, the sender resends the unacknowledged data.\n\n"
        "Flow control stops a fast sender from overwhelming a slow receiver: the "
        "receiver advertises a window saying how many more bytes it can currently "
        "buffer, and the sender never has more than that outstanding. Congestion "
        "control, which is separate, stops senders from overwhelming the network: a "
        "connection starts with a small congestion window and grows it as "
        "acknowledgements return, backing off sharply when loss is detected. A "
        "connection is closed with a FIN from each side, each FIN being acknowledged.",
    ),
    (
        "8. The Domain Name System",
        "DNS translates human-readable names such as example.com into IP addresses. The "
        "namespace is a hierarchy: the root servers delegate to top-level domain "
        "servers such as .com, which delegate to the authoritative servers for an "
        "individual domain. A client normally asks a recursive resolver, which walks "
        "that hierarchy on the client's behalf and returns the final answer.\n\n"
        "Common record types: an A record maps a name to an IPv4 address; an AAAA "
        "record maps a name to an IPv6 address; a CNAME record makes one name an alias "
        "for another; an MX record names the mail servers for a domain; NS records name "
        "a zone's authoritative servers; and a TXT record holds arbitrary text. Every "
        "record carries a time-to-live that tells resolvers how many seconds they may "
        "cache it before asking again.",
    ),
    (
        "9. HTTP",
        "HTTP is the request/response protocol of the web. A request has a method, a "
        "path, headers, and an optional body. GET retrieves a resource, POST submits "
        "data, PUT replaces a resource, DELETE removes one, and HEAD asks for the "
        "headers a GET would return without the body.\n\n"
        "Every response carries a three-digit status code. The first digit gives the "
        "class: 1xx informational, 2xx success, 3xx redirection, 4xx client error (the "
        "request was malformed or not allowed, as with 404), and 5xx server error (the "
        "server failed to fulfil a valid request). HTTP is stateless, so servers use "
        "cookies to recognise repeat visitors. HTTP/1.1 keep-alive reuses one TCP "
        "connection for several sequential requests; HTTP/2 goes further and "
        "multiplexes many concurrent requests and responses over a single connection.",
    ),
    (
        "10. TLS and a full page load",
        "TLS wraps an ordinary TCP connection to give it three properties: "
        "confidentiality, so an eavesdropper learns nothing; integrity, so tampering "
        "is detected; and authentication, so the client knows which server it is "
        "talking to. The handshake uses asymmetric cryptography, together with the "
        "server's certificate, only to authenticate the server and agree on a shared "
        "symmetric session key; the much faster symmetric key then encrypts the actual "
        "application data. A certificate is trusted because it is signed by a "
        "certificate authority whose own certificate chains up to one already trusted "
        "by the client.\n\n"
        "Putting it together, loading https://example.com/ means: resolve the name via "
        "DNS; open a TCP connection with the three-way handshake; run the TLS handshake "
        "over it; send an HTTP GET for '/'; receive a 200 response whose body is HTML; "
        "and repeat the process for each stylesheet, script, and image the page "
        "references.",
    ),
]

# ======================================================================
# 2. thermostat_manual.pdf  -- synthetic product manual
# ======================================================================
MANUAL_TITLE = "Aurora T-200 Smart Thermostat - User Guide"
MANUAL_PAGES: list[tuple[str, str]] = [
    (
        "1. Welcome and safety",
        "Thank you for choosing the Aurora T-200 Smart Thermostat. This guide covers "
        "installation, setup, everyday use, and troubleshooting.\n\n"
        "In the box: the T-200 thermostat, the wall mounting plate, four mounting "
        "screws with drywall anchors, two AAA alkaline batteries, a sheet of wire "
        "labels, and a quick-start card.\n\n"
        "SAFETY. Always switch off power to your heating and cooling equipment at the "
        "breaker before removing your old thermostat or wiring the T-200. The T-200 is "
        "a low-voltage control only: it must not be connected to a 120 V or 240 V "
        "line-voltage system, such as electric baseboard heat. If you are unsure what "
        "system you have, stop and consult a licensed HVAC technician. Do not install "
        "the thermostat in direct sunlight or above a heat source, as this will bias "
        "the temperature reading.",
    ),
    (
        "2. Specifications",
        "Primary power: 24 VAC, supplied through the common (C) wire terminal. A C "
        "wire is required; the T-200 does not support power stealing.\n"
        "Backup power: two AAA alkaline batteries, which preserve the clock and "
        "settings during a power loss.\n"
        "Wire terminals: R and Rc (power), C (common), W and W2 (heat stages), Y and "
        "Y2 (cool stages), G (fan).\n"
        "Setpoint range: 40 to 90 degrees Fahrenheit, adjustable in 1-degree steps.\n"
        "Temperature sensor accuracy: plus or minus 0.5 degrees Fahrenheit.\n"
        "Wi-Fi: 2.4 GHz 802.11 b/g/n only. The T-200 does not connect to 5 GHz "
        "networks.\n"
        "Dimensions: 3.5 x 3.5 x 1.0 inches.\n"
        "Operating environment: 32 to 104 degrees Fahrenheit, 5 to 90 percent relative "
        "humidity, non-condensing.",
    ),
    (
        "3. Installation, part 1: removing the old thermostat",
        "1. Switch off power to the heating and cooling system at the breaker panel. "
        "Confirm the old thermostat's display is blank before continuing.\n"
        "2. Remove the old thermostat's face from its base. Take a photo of the wiring "
        "so you have a reference.\n"
        "3. Using the included labels, label each wire with the letter printed next to "
        "the terminal it currently occupies, not the colour of the wire.\n"
        "4. Loosen the terminal screws, release the wires, and let them rest so they "
        "cannot fall back into the wall.\n"
        "5. Unscrew and remove the old base plate from the wall.",
    ),
    (
        "4. Installation, part 2: mounting the T-200",
        "1. Feed the wires through the opening in the new wall plate. Hold the plate "
        "against the wall, use the built-in bubble level to set it straight, and mark "
        "the screw holes.\n"
        "2. Drill the marked holes, tap in the supplied anchors, and fasten the plate "
        "with the four screws.\n"
        "3. Insert each labelled wire into the matching lettered terminal. Press the "
        "tab, insert the wire until it seats, and release. Tug gently to confirm it is "
        "held.\n"
        "4. Align the thermostat body with the plate and press until it clicks.\n"
        "5. Restore power at the breaker. The screen should light within ten seconds.",
    ),
    (
        "5. First-time setup",
        "When the T-200 powers on for the first time it walks you through setup on "
        "screen.\n\n"
        "1. Choose a display language.\n"
        "2. Select your home Wi-Fi network and enter its password. Remember that only "
        "a 2.4 GHz network will appear in the list; if you run a combined network, you "
        "may need to enable the 2.4 GHz band or use a separate network name for it.\n"
        "3. Open the Aurora Home app on your phone and either sign in or create an "
        "account, then scan the pairing code shown on the thermostat.\n"
        "4. Confirm your date, time, and location so that schedules and daylight "
        "saving adjust automatically.",
    ),
    (
        "6. Everyday use",
        "Setting the temperature: turn the outer dial, or tap the on-screen up and "
        "down arrows, to change the target temperature. The large number is the "
        "current room temperature; the smaller number is the target.\n\n"
        "System mode: choose Heat, Cool, Auto, or Off. In Auto the thermostat switches "
        "between heating and cooling to keep the room between a low and a high target.\n\n"
        "Fan mode: Auto runs the fan only during a heating or cooling call; On runs it "
        "continuously; Circulate runs it for a share of each hour to even out "
        "temperatures.\n\n"
        "Hold: adjusting the target by hand starts a temporary Hold that keeps that "
        "setting until the next scheduled period. A permanent Hold ignores the "
        "schedule until you cancel it.",
    ),
    (
        "7. Schedules",
        "A schedule lets the T-200 change target temperatures automatically through "
        "the day. Each day has up to four periods, named Wake, Away, Home, and Sleep, "
        "and you set a start time and a heat and cool target for each.\n\n"
        "Weekdays and weekends are scheduled separately. After editing one day you can "
        "use 'Copy to' to duplicate its periods onto other days rather than re-entering "
        "them.\n\n"
        "If you make a manual change while a schedule is running, the resulting "
        "temporary Hold lasts only until the next period begins, at which point the "
        "schedule resumes on its own.",
    ),
    (
        "8. Energy-saving features",
        "Eco mode: when enabled, Eco mode widens both the heat and cool targets by 4 "
        "degrees from their scheduled values to cut runtime while you are asleep or "
        "out.\n\n"
        "Adaptive Recovery: the thermostat learns how long your system takes to change "
        "the room temperature and starts heating or cooling early, so the room reaches "
        "the scheduled target at the scheduled time rather than starting from it.\n\n"
        "Away detection: if you allow location access, the Aurora Home app tells the "
        "thermostat when everyone's phone has left a geofence around the house so it "
        "can drop to an economical setting, and to resume normal targets before you "
        "get home.\n\n"
        "Energy report: each month the app summarises heating and cooling runtime and "
        "compares it with the previous month.",
    ),
    (
        "9. Maintenance and troubleshooting",
        "Maintenance: replace the two AAA batteries once a year, or sooner if the "
        "low-battery icon appears. Clean the screen with a dry, soft cloth only; do "
        "not use liquids or sprays. After changing a furnace filter, reset the "
        "reminder under Settings > Maintenance.\n\n"
        "Screen is blank: the system has no power. Check that the breaker is on and "
        "that a C wire is connected to the C terminal; check the AAA batteries.\n"
        "Will not join Wi-Fi: confirm you are joining a 2.4 GHz network and that the "
        "password is correct; move the router closer or add an extender.\n"
        "Heat will not start: check the wire in the W terminal; note that a built-in "
        "five-minute compressor delay can postpone the start of a call.\n"
        "System short-cycles: raise the cycle-rate setting so the equipment runs "
        "longer per cycle.\n"
        "Temperature reading looks high: make sure the thermostat is out of direct sun "
        "and away from lamps and electronics.",
    ),
    (
        "10. Warranty and support",
        "Limited warranty: Aurora warrants the T-200 against defects in materials and "
        "workmanship for three years from the date of purchase. The warranty does not "
        "cover damage from improper installation, misuse, or connection to a "
        "line-voltage system.\n\n"
        "FCC: this device complies with Part 15 of the FCC Rules. Operation is subject "
        "to the condition that this device may not cause harmful interference and must "
        "accept any interference received.\n\n"
        "Support: reach Aurora support at 1-800-555-0142 or support@example.com, "
        "Monday to Friday, 8am to 8pm Eastern.\n\n"
        "Factory reset: to erase all settings and Wi-Fi credentials, press and hold "
        "the dial for ten seconds until the screen shows RESET, then confirm. The "
        "thermostat restarts into first-time setup.",
    ),
]


def _fits(paragraphs: list[str]) -> bool:
    """True if these paragraphs fit in one body text box."""
    scratch = fitz.open()
    page = scratch.new_page(width=PAGE_W, height=PAGE_H)
    leftover = page.insert_textbox(
        TEXT_RECT,
        "\n\n".join(paragraphs),
        fontsize=BODY_FONTSIZE,
        fontname="helv",
        lineheight=BODY_LEADING / BODY_FONTSIZE,
    )
    scratch.close()
    return leftover >= 0


def _typeset_license(txt_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    """Lay out the verbatim Apache-2.0 text, flowing paragraphs onto as
    many pages as needed rather than guessing page breaks by hand.
    """
    raw = txt_path.read_text(encoding="utf-8")

    # The plain-text license indents every line with leading spaces and
    # hard-wraps within paragraphs. Strip the indent and join hard-wrapped
    # lines back into single paragraphs, treating a blank line as the
    # paragraph separator.
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            buf.append(stripped)
        elif buf:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    # Greedy pagination: keep adding whole paragraphs to the current page
    # until either the text box would overflow or the page is already
    # fairly full (SOFT_CHARS), then start a new page. The soft cap just
    # keeps pages from being wall-to-wall text, so citations resolve to a
    # smaller span -- it doesn't change the wording.
    SOFT_CHARS = 1500
    pages: list[tuple[str, str]] = []
    current: list[str] = []
    for para in paragraphs:
        too_full = sum(len(p) for p in current) >= SOFT_CHARS
        if current and (too_full or not _fits(current + [para])):
            pages.append(("", "\n\n".join(current)))
            current = []
        current.append(para)
    if current:
        pages.append(("", "\n\n".join(current)))

    heading = "Apache License, Version 2.0"
    pages = [
        (heading if i == 0 else f"{heading} (p. {i + 1})", body)
        for i, (_, body) in enumerate(pages)
    ]
    _write_pages(pdf_path, heading, pages)


def build() -> None:
    _write_pages(HERE / "notes_primer.pdf", NOTES_TITLE, NOTES_PAGES)
    _write_pages(HERE / "thermostat_manual.pdf", MANUAL_TITLE, MANUAL_PAGES)
    _typeset_license(HERE / "apache_license_2.0.txt", HERE / "apache_license_2.0.pdf")


if __name__ == "__main__":
    build()
