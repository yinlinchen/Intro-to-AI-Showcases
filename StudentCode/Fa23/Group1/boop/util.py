# Utilities, mostly for testing, but useful all around.

def list_match(expected: list, actual: list):
    """
    Test helper for matching sets of pts, as we don't specify order.
    """
    if len(expected) != len(actual):
        return False
    for pt in expected:
        if not pt in actual:
            return False
    return True


def any_match(lst_a: list, lst_of_lsts: list[list]):
    """
    Test helper for matching a list against a set of lists, where
    order doesn't matter
    """
    for lst in lst_of_lsts:
        if list_match(lst_a, lst):
            return True
    return False


def all_have_match(expected: list[list], actual: list[list]):
    """
    # Test helper for matching a list of lists against a list of lists, where
    # order doesn't matter
    """
    if len(actual) != len(expected):
        return False
    for lst in expected:
        if not any_match(lst, actual):
            print("no match for: ", lst)
            return False
    return True

