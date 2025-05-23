import re
from config.config import SURAH_NAME_TO_NO

VERSE_SPEC_REGEX = re.compile(
    r"""
    (?:([\w-]+)\s*(?:\(|verse|ayah|ayat))? # Optional Surah Name (group 1) followed by ( or verse/ayah
    \s*(\d+):(\d+)                        # SurahNo (group 2) : AyahStart (group 3)
    (?:[-–—]\s*(\d+))?                    # Optional - AyahEnd (group 4)
    \s*\)?                                # Optional closing parenthesis
    """,
    re.IGNORECASE | re.VERBOSE
)

ORIGINAL_ID_REGEX = re.compile(
    r"(?:verse|ayah|ayat)\s+(\d+).*?(?:surah|chapter)\s+([\w-]+)",
    re.IGNORECASE
)

def parse_verse_references(query_text: str) -> list[dict]:
    """
    Parses a query text to find all Surah:Ayah references, including ranges.
    Returns a list of dictionaries, each specifying surah_no, ayah_start, ayah_end.
    """
    references = []

    for match in VERSE_SPEC_REGEX.finditer(query_text):
        surah_name_match = match.group(1)
        surah_no_str = match.group(2)
        ayah_start_str = match.group(3)
        ayah_end_str = match.group(4)

        surah_no = -1
        if surah_name_match:
            surah_name_input = surah_name_match.strip().lower()
            surah_no_from_name = SURAH_NAME_TO_NO.get(surah_name_input, -1)

            if surah_no_from_name != -1:
                if surah_no_from_name != int(surah_no_str):
                    print(f"Warning: Ambiguous Surah in '{match.group(0)}'. Name '{surah_name_input}' (Surah {surah_no_from_name}) vs. ':{surah_no_str}'. Using ':{surah_no_str}'.")
                    surah_no = int(surah_no_str)
                else:
                    surah_no = surah_no_from_name
            else: # Name not found in map, but number is present
                surah_no = int(surah_no_str)
        elif surah_no_str: # No surah name, just S:A
             surah_no = int(surah_no_str)

        if surah_no == -1:
            print(f"Could not determine Surah number for match: '{match.group(0)}'")
            continue

        try:
            ayah_start = int(ayah_start_str)
            ayah_end = int(ayah_end_str) if ayah_end_str else ayah_start

            if ayah_end < ayah_start:
                print(f"Warning: Ayah end ({ayah_end}) is less than Ayah start ({ayah_start}) in '{match.group(0)}'. Treating as single verse {ayah_start}.")
                ayah_end = ayah_start

            references.append({
                "surah_no": surah_no,
                "ayah_start": ayah_start,
                "ayah_end": ayah_end,
                "original_match": match.group(0)
            })
        except ValueError:
            print(f"Warning: Could not parse Ayah numbers in '{match.group(0)}'.")
            continue

    # Fallback for "verse X surah Y" format if primary regex found nothing
    if not references:
        for match in ORIGINAL_ID_REGEX.finditer(query_text):
            try:
                ayah_no = int(match.group(1))
                surah_name_input = match.group(2).lower()
                surah_no = SURAH_NAME_TO_NO.get(surah_name_input, -1)

                if surah_no != -1:
                    references.append({
                        "surah_no": surah_no,
                        "ayah_start": ayah_no,
                        "ayah_end": ayah_no,
                        "original_match": match.group(0)
                    })
                else:
                    print(f"Warning: Surah name '{surah_name_input}' not found in map for match '{match.group(0)}'.")
            except ValueError:
                print(f"Warning: Could not parse Ayah number in ORIGINAL_ID_REGEX match '{match.group(0)}'.")
                continue
    
    # Deduplicate references
    if references:
        unique_references = []
        seen_tuples = set()
        for ref in references:
            ref_tuple = (ref["surah_no"], ref["ayah_start"], ref["ayah_end"])
            if ref_tuple not in seen_tuples:
                unique_references.append(ref)
                seen_tuples.add(ref_tuple)
        references = unique_references
        
    return references

def route_query(query_text: str):
    """
    Routes the query to 'exact_multiple' if specific verse patterns are found,
    otherwise defaults to 'rag' mode.
    """
    parsed_references = parse_verse_references(query_text)

    if parsed_references:
        return "exact_multiple", {"references": parsed_references}
    return "rag", {}
