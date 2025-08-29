
def compare_mappings(old_mapping, new_mapping):
    """
    Compares two dictionaries and identifies:
    - Unchanged mappings
    - Changed mappings
    - Newly added mappings
    - Removed mappings
    """
    unchanged = {}
    changed = {}
    added = {}
    removed = {}

    for key in old_mapping:
        if key in new_mapping:
            if old_mapping[key] == new_mapping[key]:
                unchanged[key] = old_mapping[key]
            else:
                changed[key] = (old_mapping[key], new_mapping[key])
        else:
            removed[key] = old_mapping[key]

    for key in new_mapping:
        if key not in old_mapping:
            added[key] = new_mapping[key]

    return {"unchanged": unchanged, "changed": changed, "added": added, "removed": removed}