import random
def gen_ints(num, range, delta):

    values = []

    while len(values) < num:
        candidate = random.randrange(*range)
        is_candidate_valid = True

        for val in values:

            if abs(candidate - val) < delta:
                is_candidate_valid = False
                break

        if is_candidate_valid:
            values.append(candidate)

    return values