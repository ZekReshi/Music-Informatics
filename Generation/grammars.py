import random as rd


ONSET = 'O'
HOLD = '-'
MUTE = 'X'


rules = {
    "FullBar": [
        {
            'slow': [0.3, 0.2, 0.5, 0, 0],
            'normal': [0.15, 0.3, 0.3, 0.25, 0],
            'fast': [0, 0.3, 0, 0.3, 0.4]
        },
        ["Whole"],
        ["HalfBar", "HalfBar"],
        ["SlowHalfBar", "HalfBar"],
        ["HalfBar", "FastHalfBar"],
        ["FastHalfBar", "FastHalfBar"]
    ],
    "HalfBar": [
        {
            'slow': [1, 0, 0],
            'normal': [0.4, 0.3, 0.3],
            'fast': [0.1, 0.45, 0.45]
        },
        ["Quarter", "Quarter"],
        ["Quarter", "Eighth", "Eighth"],
        ["Eighth", "Eighth", "Quarter"]
    ],
    "SlowHalfBar": [
        {
            'slow': [0.6, 0.2, 0.2],
            'normal': [0.4, 0.3, 0.3],
            'fast': [0.2, 0.4, 0.4]
        },
        ["Half"],
        ["QuarterExt", "Eighth"],
        ["Eighth", "QuarterExt"]
    ],
    "FastHalfBar": [
        {
            'slow': [1, 0, 0, 0, 0],
            'normal': [0.5, 0.125, 0.125, 0.125, 0.125],
            'fast': [0.1, 0.225, 0.225, 0.225, 0.225]
        },
        ["Eighth", "Eighth", "Eighth", "Eighth"],
        ["Sixteenth", "Sixteenth", "Eighth", "Eighth", "Eighth"],
        ["Eighth", "Sixteenth", "Sixteenth", "Eighth", "Eighth"],
        ["Eighth", "Eighth", "Sixteenth", "Sixteenth", "Eighth"],
        ["Eighth", "Eighth", "Eighth", "Sixteenth", "Sixteenth"]
    ],
    "Whole": [
        [1],
        [ONSET] + [HOLD for _ in range(15)]
    ],
    "Half": [
        [1],
        [ONSET] + [HOLD for _ in range(7)]
    ],
    "Quarter": [
        [0.9, 0.1],
        [ONSET] + [HOLD for _ in range(3)],
        [MUTE for _ in range(4)]
    ],
    "QuarterExt": [
        [0.9, 0.1],
        [ONSET] + [HOLD for _ in range(5)],
        [MUTE for _ in range(6)]
    ],
    "Eighth": [
        [0.85, 0.10],
        [ONSET + HOLD],
        [MUTE + MUTE]
    ],
    "Sixteenth": [
        [0.9, 0.1],
        [ONSET],
        [MUTE]
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


def expansion(start, speed: str = 'normal'):
    for element in start:
        if element in rules:
            loc = start.index(element)
            weights = rules[element][0]
            start[loc] = rd.choices(rules[element][1:], weights=weights if type(weights) is list else weights[speed])
        result = [item for item in generate_items(start)]

    for item in result:
        if not isinstance(item, list):
            if item in rules:
                result = expansion(result)

    return result


def get_rhythm(bars: int = 8, speed: str = 'normal', double_bars: bool = True):
    rhythm = ""
    for i in range(bars if not double_bars else bars // 2):
        bar = expansion(["FullBar"], speed=speed)
        bar_str = to_string(bar)
        rhythm += bar_str + bar_str if double_bars and i < bars-1 else bar_str

    return rhythm


def to_string(result):
    return ''.join(result)


if __name__ == '__main__':
    print(get_rhythm())
