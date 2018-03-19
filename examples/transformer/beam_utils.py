class Hypothesis:
    def __init__(self, score, output, state):
        self.score = score
        self.state = state
        self.output = output

def default_len_normalize_partial(score_so_far, score_to_add, new_len):
    return score_so_far + score_to_add
