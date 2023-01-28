import re


class LicensePattern:
    cc_pattern = re.compile("http[s]?://creativecommons\\.org/licenses/(by|by-sa|by-nd|by-nc|by-nc-sa|by-nc-nd|publicdomain)[\"/ >]")

def detect_licence(html:str):
    """
    Given a HTML string, this function detects the licence of the page.
    It returns a string with the licence name, or NO-LICENCE-FOUND if no licence is found.
    """
    license_attribute_pattern = re.compile(LicensePattern.cc_pattern)

    # storing counts of all difference occurrences of link to CC
    multiple_occurrences_map = {}

    # add all of them to the list
    for match in license_attribute_pattern.finditer(html):
        licence = match.group(1)

        # add entry
        if licence not in multiple_occurrences_map:
            multiple_occurrences_map[licence] = 0

        # and increase count
        multiple_occurrences_map[licence] += 1

    # no licence found
    if not multiple_occurrences_map:
        return "no-licence-found"

    # only one link found or if multiple links found but the same type
    if len(multiple_occurrences_map) == 1:
        return list(multiple_occurrences_map.keys())[0]

    # if multiple different links found, we return a general CC-UNSPECIFIED
    return "cc-unspecified"


