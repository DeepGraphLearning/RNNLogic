import ast


def input_choice(prompt, choice=("y", "n")):
    prompt = "%s (%s)" % (prompt, "/".join(choice))
    choice = set([c.lower() for c in choice])
    result = input(prompt)
    while result.lower() not in choice:
        result = input(prompt)
    return result


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string