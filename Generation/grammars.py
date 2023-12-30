import random as rd

rules = {
    "FullBar": [
        [0.15, 0.3, 0.3, 0.25],
        ["Whole"],
        ["HalfBar", "HalfBar"],
        ["SlowHalfBar", "HalfBar"],
        ["HalfBar", "FastHalfBar"]
    ],
    "HalfBar": [
        [0.4, 0.3, 0.3],
        ["Quarter", "Quarter"],
        ["Quarter", "Eighth", "Eighth"],
        ["Eighth", "Eighth", "Quarter"]
    ],
    "SlowHalfBar": [
        [0.4, 0.3, 0.3],
        ["Half"],
        ["QuarterExt", "Eighth"],
        ["Eighth", "QuarterExt"]
    ],
    "FastHalfBar": [
        [0.5, 0.125, 0.125, 0.125, 0.125],
        ["Eighth", "Eighth", "Eighth", "Eighth"],
        ["Sixteenth", "Sixteenth", "Eighth", "Eighth", "Eighth"],
        ["Eighth", "Sixteenth", "Sixteenth", "Eighth", "Eighth"],
        ["Eighth", "Eighth", "Sixteenth", "Sixteenth", "Eighth"],
        ["Eighth", "Eighth", "Eighth", "Sixteenth", "Sixteenth"]
    ],
    "Whole": [
        [1],
        ["O---------------"]
    ],
    "Half": [
        [0.85, 0.15],
        ["O-------"],
        ["XXXXXXXX"]
    ],
    "Quarter": [
        [0.85, 0.15],
        ["O---"],
        ["XXXX"]
    ],
    "QuarterExt": [
        [0.85, 0.15],
        ["O-----"],
        ["XXXXXX"]
    ],
    "Eighth": [
        [0.85, 0.15],
        ["O-"],
        ["XX"]
    ],
    "Sixteenth": [
        [0.85, 0.15],
        ["O"],
        ["X"]
    ]
}


# Contributed by Tiger Sachase
# Used to parse any list of strings and insert them in place in a list
def generate_items(items):
    for item in items:
        if isinstance(item, list):
            for subitem in generate_items(item):
                yield subitem
        else:
            yield item


def expansion(start):
    for element in start:
        if element in rules:
            loc = start.index(element)
            start[loc] = rd.choices(rules[element][1:], weights=rules[element][0])
        result = [item for item in generate_items(start)]

    for item in result:
        if not isinstance(item, list):
            if item in rules:
                result = expansion(result)

    return result


def to_string(result):
    return ''.join(result)


if __name__ == '__main__':
    for _ in range(8):
        result = expansion(["FullBar"])  # Expand our starting list
        final = to_string(result)
        print(final)  # Print the final result
