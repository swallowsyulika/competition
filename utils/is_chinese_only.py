import re

ignore_pat = re.compile("[A-z0-9!@#$%^&*()_-]+")

def is_chinese_only(pattern: str):
    if ignore_pat.search(pattern):
        return False
    else:
        return True